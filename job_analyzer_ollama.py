import os
import json
import duckdb
import pandas as pd
import graphviz
import argparse
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # Default to llama3 if not specified
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # Default to local Ollama
DB_PATH = os.path.join("database", "jobs_database.db")

class JobAnalyzer:
    def __init__(self, db_path=DB_PATH, knowledge_dir="knowledge_articles/", 
                 ollama_model=OLLAMA_MODEL, ollama_host=OLLAMA_HOST):
        self.db_path = db_path
        self.knowledge_dir = knowledge_dir
        self.db_conn = None
        self.vector_store = None
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        
        # Initialize LLM
        self.llm = Ollama(model=ollama_model, base_url=ollama_host)
        
        # Setup database and vector store
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
                embedding_function=OllamaEmbeddings(model=self.ollama_model, base_url=self.ollama_host),
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
                            "page_content": content,
                            "metadata": {"source": filename}
                        })
        
        # Split documents into chunks
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = []
            for doc in documents:
                chunks = text_splitter.split_text(doc["page_content"])
                for chunk in chunks:
                    docs.append({
                        "page_content": chunk,
                        "metadata": {"source": doc["metadata"]["source"]}
                    })
            
            # Create vector store with Ollama embeddings
            self.vector_store = Chroma.from_documents(
                documents=docs,
                embedding=OllamaEmbeddings(model=self.ollama_model, base_url=self.ollama_host),
                collection_name="knowledge_articles",
                persist_directory="./chroma_db"
            )
            self.vector_store.persist()
        else:
            print("No documents found in knowledge directory")
        
    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()

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
        predecessors = result["predecessor_job"].dropna().tolist() if "predecessor_job" in result.columns else []
        successors = result["successor_job"].dropna().tolist() if "successor_job" in result.columns else []
        file_triggers = result["file_trigger"].dropna().tolist() if "file_trigger" in result.columns else []
        
        response = {
            "job_name": job_name,
            "predecessors": predecessors,
            "successors": successors,
            "file_triggers": file_triggers,
            "total_predecessors": len(predecessors),
            "total_successors": len(successors)
        }
        
        return json.dumps(response, indent=2)

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
        parents = result[result["child_job"] == job_name]["parent_job"].tolist() if "parent_job" in result.columns and "child_job" in result.columns else []
        children = result[result["parent_job"] == job_name]["child_job"].tolist() if "parent_job" in result.columns and "child_job" in result.columns else []
        
        response = {
            "job_name": job_name,
            "parents": parents,
            "children": children,
            "total_parents": len(parents),
            "total_children": len(children)
        }
        
        return json.dumps(response, indent=2)

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
                
                if "predecessor_job" in result.columns:
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
                
                if "successor_job" in result.columns:
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
        Tool(
            name="get_job_dependencies",
            func=job_analyzer.get_job_dependencies,
            description="Retrieve the predecessor and successor dependencies for a given job. Input should be a job name."
        ),
        Tool(
            name="get_job_criticality",
            func=job_analyzer.get_job_criticality,
            description="Retrieve information about job criticality, SLAs, and business impact. Input should be a job name."
        ),
        Tool(
            name="get_job_parent_child_relations",
            func=job_analyzer.get_job_parent_child_relations,
            description="Retrieve parent-child relationships for a job from the adjacency list. Input should be a job name."
        ),
        Tool(
            name="search_knowledge_articles",
            func=job_analyzer.search_knowledge_articles,
            description="Search knowledge articles for information related to job failures and resolutions. Input should be a search query."
        ),
        Tool(
            name="generate_dependency_graph",
            func=job_analyzer.generate_dependency_graph,
            description="Generate a GraphViz visualization of the job's dependencies. Input should be a job name."
        )
    ]
    
    # Create prompt with ReAct framework
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant specialized in analyzing mainframe jobs.

For non-job queries like greetings, IMMEDIATELY respond with a simple greeting.
Example: If user says "hello", just respond "Hello! How can I help you analyze mainframe jobs today?"

For job-related queries, you MUST follow this EXACT format:
Thought: [brief thought about what to do next]
Action: [name of the tool to use]
Action Input: [exact input for the tool]
Observation: [wait for tool response]
Thought: [brief thought about the result and what to do next]
... (repeat if needed)
Final Answer: [summarize findings for the user]

Example format for job query:
Thought: I need to check job dependencies first
Action: get_job_dependencies
Action Input: "JOB123"
Observation: [tool response]                    
Thought: Now I need to check criticality
Action: get_job_criticality
Action Input: "JOB123"
Observation: [tool response]
Final Answer: Based on the analysis, JOB123...

Available tools:
{tools}

Tool Names: {tool_names}"""),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])
    
    # Create agent with ReAct formatting
    agent = create_react_agent(
        llm=job_analyzer.llm, 
        tools=tools, 
        prompt=prompt,
    )
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor

# Streamlit UI
def create_streamlit_app():
    st.title("Job Analysis Assistant")
    st.write("Ask questions about job dependencies, criticality, and troubleshooting.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        model = st.selectbox(
            "Select Ollama Model", 
            ["llama3", "mistral", "llama2", "llama3:8b", "codellama", "phi3", "deepseek-r1:1.5b"], 
            index=0
        )
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    
    # Initialize analyzer with selected model
    job_analyzer = JobAnalyzer(ollama_model=model, ollama_host=ollama_url)
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
                try:
                    response = agent.invoke({"input": user_query})
                    assistant_response = response["output"]
                    st.write(assistant_response)
                    
                    # Check if a graph was generated
                    if "Dependency graph generated" in assistant_response:
                        # Extract job name from response
                        import re
                        match = re.search(r'graphs/(\w+)_dependencies\.png', assistant_response)
                        if match:
                            job_name = match.group(1)
                            st.image(f"graphs/{job_name}_dependencies.png")
                except Exception as e:
                    st.error(f"Error processing your request: {str(e)}")
                    assistant_response = "I encountered an error while processing your request. Please try again or rephrase your question."
        
        # Add assistant message to chat
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Command line interface
def cli():
    parser = argparse.ArgumentParser(description='Job Analysis CLI')
    parser.add_argument('--query', type=str, required=True, help='Query about a job')
    parser.add_argument('--model', type=str, default=OLLAMA_MODEL, help='Ollama model to use')
    parser.add_argument('--host', type=str, default=OLLAMA_HOST, help='Ollama host URL')
    args = parser.parse_args()
    
    job_analyzer = JobAnalyzer(ollama_model=args.model, ollama_host=args.host)
    agent = create_agent(job_analyzer)
    
    try:
        print(f"Processing query: {args.query}")
        print(f"Using model: {args.model} at {args.host}")
        response = agent.invoke({"input": args.query})
        print("\nResponse:")
        print(response["output"])
    finally:
        job_analyzer.close()

# Main entry point
if __name__ == "__main__":
    # Check if running in Streamlit
    # print(f"{os.environ=}")
    # if 'STREAMLIT_RUN_ENV' in os.environ:
    create_streamlit_app()
    # else:
        # cli()