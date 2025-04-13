import os
import json
import duckdb
import pandas as pd
import graphviz
import argparse
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool, tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable")

# Initialize database connection
DB_PATH = os.getenv("DUCKDB_PATH", "jobs_database.db")

class JobAnalyzer:
    def __init__(self, db_path=DB_PATH, knowledge_dir="knowledge_articles/"):
        self.db_path = db_path
        self.knowledge_dir = knowledge_dir
        self.db_conn = None
        self.vector_store = None
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.setup()
        
    def setup(self):
        """Set up database connection and vector store"""
        # Connect to DuckDB
        self.db_conn = duckdb.connect(self.db_path)
        
        # Initialize vector store for knowledge articles if not already done
        if not os.path.exists("./chroma_db"):
            self._initialize_vector_store()
        else:
            self.vector_store = Chroma(
                collection_name="knowledge_articles",
                embedding_function=OpenAIEmbeddings(),
                persist_directory="./chroma_db"
            )
    
    def _initialize_vector_store(self):
        """Process knowledge articles and create vector store"""
        documents = []
        
        # Load knowledge articles
        if os.path.exists(self.knowledge_dir):
            for filename in os.listdir(self.knowledge_dir):
                if filename.endswith('.txt') or filename.endswith('.md'):
                    file_path = os.path.join(self.knowledge_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append({
                            "content": content,
                            "source": filename
                        })
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = []
        for doc in documents:
            chunks = text_splitter.split_text(doc["content"])
            for chunk in chunks:
                docs.append({
                    "page_content": chunk,
                    "metadata": {"source": doc["source"]}
                })
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=docs,
            embedding=OpenAIEmbeddings(),
            collection_name="knowledge_articles",
            persist_directory="./chroma_db"
        )
        self.vector_store.persist()
        
    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()

    @tool("get_job_dependencies")
    def get_job_dependencies(self, job_name: str) -> str:
        """Retrieve the predecessor and successor dependencies for a given job"""
        query = f"""
        SELECT * FROM mainframe_deps 
        WHERE job_name = '{job_name}'
        """
        result = self.db_conn.execute(query).fetchdf()
        
        if result.empty:
            return f"No dependency information found for job '{job_name}'"
            
        # Format dependencies
        predecessors = result["predecessor_job"].dropna().tolist()
        successors = result["successor_job"].dropna().tolist()
        file_triggers = result["file_trigger"].dropna().tolist()
        
        response = {
            "job_name": job_name,
            "predecessors": predecessors,
            "successors": successors,
            "file_triggers": file_triggers,
            "total_predecessors": len(predecessors),
            "total_successors": len(successors)
        }
        
        return json.dumps(response, indent=2)

    @tool("get_job_parent_child_relations")
    def get_job_parent_child_relations(self, job_name: str) -> str:
        """Retrieve parent-child relationships for a job from the adjacency list"""
        query = f"""
        SELECT * FROM adgecency_list
        WHERE parent_job = '{job_name}' OR child_job = '{job_name}'
        """
        result = self.db_conn.execute(query).fetchdf()
        
        if result.empty:
            return f"No parent-child relationships found for job '{job_name}'"
            
        # Get parents and children
        parents = result[result["child_job"] == job_name]["parent_job"].tolist()
        children = result[result["parent_job"] == job_name]["child_job"].tolist()
        
        response = {
            "job_name": job_name,
            "parents": parents,
            "children": children,
            "total_parents": len(parents),
            "total_children": len(children)
        }
        
        return json.dumps(response, indent=2)

    @tool("get_job_criticality")
    def get_job_criticality(self, job_name: str) -> str:
        """Retrieve information about job criticality, SLAs, and business impact"""
        query = f"""
        SELECT * FROM mainframe_critical_jobs
        WHERE job_name = '{job_name}'
        """
        result = self.db_conn.execute(query).fetchdf()
        
        if result.empty:
            return f"Job '{job_name}' is not listed as a critical job"
            
        # Convert to dict for JSON serialization
        job_info = result.iloc[0].to_dict()
        
        return json.dumps(job_info, indent=2)

    @tool("search_knowledge_articles")
    def search_knowledge_articles(self, query: str) -> str:
        """Search knowledge articles for information related to job failures and resolutions"""
        if not self.vector_store:
            return "Knowledge base not initialized"
            
        # Use LLM to extract relevant context
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
        )
        
        # Retrieve relevant documents
        docs = compression_retriever.get_relevant_documents(query)
        
        if not docs:
            return "No relevant knowledge articles found"
            
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "article_number": i+1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown")
            })
            
        return json.dumps(results, indent=2)

    @tool("generate_dependency_graph")
    def generate_dependency_graph(self, job_name: str, depth: int = 1) -> str:
        """Generate a GraphViz visualization of the job's dependencies"""
        # Create a Graphviz graph
        dot = graphviz.Digraph(comment=f'Dependencies for {job_name}')
        
        # Add the main job
        dot.node(job_name, style='filled', color='lightblue')
        
        # Function to recursively add dependencies
        def add_dependencies(current_job, current_depth, direction):
            if current_depth > depth:
                return
                
            if direction == "predecessor":
                query = f"""
                SELECT predecessor_job FROM mainframe_deps 
                WHERE job_name = '{current_job}' AND predecessor_job IS NOT NULL
                """
                result = self.db_conn.execute(query).fetchdf()
                
                for _, row in result.iterrows():
                    pred_job = row['predecessor_job']
                    dot.node(pred_job)
                    dot.edge(pred_job, current_job)
                    add_dependencies(pred_job, current_depth + 1, direction)
                    
            elif direction == "successor":
                query = f"""
                SELECT successor_job FROM mainframe_deps 
                WHERE job_name = '{current_job}' AND successor_job IS NOT NULL
                """
                result = self.db_conn.execute(query).fetchdf()
                
                for _, row in result.iterrows():
                    succ_job = row['successor_job']
                    dot.node(succ_job)
                    dot.edge(current_job, succ_job)
                    add_dependencies(succ_job, current_depth + 1, direction)
        
        # Add predecessors and successors
        add_dependencies(job_name, 1, "predecessor")
        add_dependencies(job_name, 1, "successor")
        
        # Generate graph
        graph_path = f"graphs/{job_name}_dependencies"
        os.makedirs("graphs", exist_ok=True)
        dot.render(graph_path, format='png', cleanup=True)
        
        return f"Dependency graph generated and saved as {graph_path}.png"

# Create AI agent
def create_agent(job_analyzer):
    # Define tools
    tools = [
        job_analyzer.get_job_dependencies,
        job_analyzer.get_job_criticality,
        job_analyzer.get_job_parent_child_relations,
        job_analyzer.search_knowledge_articles,
        job_analyzer.generate_dependency_graph
    ]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant specialized in analyzing mainframe jobs, their dependencies, and business impacts.
Your goal is to help users understand job relationships, critical paths, and how to troubleshoot issues.

Use the available tools to retrieve information from the database and knowledge articles.
Always follow these steps:
1. Identify what information the user is requesting about jobs
2. Use appropriate tools to gather that information
3. Combine the information into a clear, concise response
4. If applicable, suggest generating a dependency graph

Format your response in a structured way but use natural language.
"""),
        ("human", "{input}"),
    ])
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

# Streamlit UI
def create_streamlit_app():
    st.title("Job Analysis Assistant")
    st.write("Ask questions about job dependencies, criticality, and troubleshooting.")
    
    job_analyzer = JobAnalyzer()
    agent = create_agent(job_analyzer)
    
    # Create chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    user_query = st.chat_input("What would you like to know about a job?")
    if user_query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        
        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.invoke({"input": user_query})
                assistant_response = response["output"]
                st.write(assistant_response)
                
                # Check if a graph was generated
                if "Dependency graph generated" in assistant_response:
                    graph_file = response.get("graph_path", "").replace("graphs/", "").replace(".png", "")
                    if graph_file:
                        st.image(f"graphs/{graph_file}.png")
        
        # Add assistant message to chat
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})


# Command line interface
def cli():
    parser = argparse.ArgumentParser(description='Job Analysis CLI')
    parser.add_argument('--query', type=str, required=True, help='Query about a job')
    args = parser.parse_args()
    
    job_analyzer = JobAnalyzer()
    agent = create_agent(job_analyzer)
    
    try:
        response = agent.invoke({"input": args.query})
        print(response["output"])
    finally:
        job_analyzer.close()

# Main entry point
if __name__ == "__main__":
    # Check if running in Streamlit
    if 'STREAMLIT_RUN_ENV' in os.environ:
        create_streamlit_app()
    else:
        cli()