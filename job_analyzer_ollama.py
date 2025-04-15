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
from langchain.llms import huggingface_hub
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder,PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document  # Add this import at the top
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage
import streamlit as st

# Add Graphviz to PATH
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Load environment variables
load_dotenv('./.env')  
# print(os.environ["HUGGINGFACEHUB_API_TOKEN"])

# Initialize Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # Default to llama3 if not specified
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # Default to local Ollama
DB_PATH = os.path.join("database", "jobs_database.db")


class JobAnalyzer:
    def __init__(self, db_path=DB_PATH, knowledge_dir="knowledge_articles/", 
                 ollama_model=OLLAMA_MODEL, ollama_host=OLLAMA_HOST,ollama_true:bool=True):
        self.db_path = db_path
        self.knowledge_dir = knowledge_dir
        self.db_conn = None
        self.vector_store = None
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        
        # Initialize LLM
        if ollama_true:
            if ollama_model.startswith("llama"):
                self.llm = Ollama(
                    model=ollama_model,
                    base_url=ollama_host,
                    temperature=0.1,  # Lower temperature for more consistent formatting
                    num_ctx=4096,     # Increase context window
                    repeat_penalty=1.2  # Reduce repetition
                    # stop=["Observation:", "Human:", "Assistant:"]  # Add explicit stop tokens
                    # stop=["Invalid Format"]
                )
            else:
                self.llm = Ollama(model=ollama_model, base_url=ollama_host)
        else:
            # Create agent with ReAct formatting
            # self.init_huggingface_model()
            self.init_google_model()
        
        # Setup database and vector store
        self.setup()
    

    def init_huggingface_model(self):
        # Choose a good model for agent use - these are some options:
        # - mistralai/Mixtral-8x7B-Instruct-v0.1  (powerful but might hit rate limits)
        # - google/gemma-7b-it                    (good balance)
        # - microsoft/phi-2                       (smaller but efficient)
        # - meta-llama/Llama-2-13b-chat-hf        (good if you've requested access)
        
        self.llm = huggingface_hub.HuggingFaceHub(
            repo_id="google/gemma-3-1b-it",  # Good starting point
            model_kwargs={
                "temperature": 0.1,
                "max_new_tokens": 512,
                "top_p": 0.95,
            }
        )


    def init_google_model(self):
        # To use the latest *1.5 Pro* model (often the most capable, might be in preview):
        model_name = "gemini-2.0-flash-lite" # Or check Google's docs for the absolute latest identifier
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            # google_api_key=api_key, # Uncomment if you didn't set the env var
            temperature=0.1,        # Example: Control creativity (0.0 - 1.0)
            max_retries=2,
            timeout=None
        )

    def setup(self):
        """Set up database connection and vector store"""
        # Connect to DuckDB
        self.db_conn = duckdb.connect(self.db_path)
        
        # Initialize vector store for knowledge articles if not already done
        if not os.path.exists("./chroma_db"):
            print("Chroma Db not found")
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
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"source": doc["metadata"]["source"]}
                    ))

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
    

    # function to greet the user, if they greet first and if they introduce themselves,retrun a greeting with their name.
    # This function is usually called when the user first interacts with the assistant
    def greet_user(self, name: str = "") -> str:
        """Greet the user with a friendly message"""
        # Handle empty input or just whitespace
        if not name or name.strip() == "" or name.lower() in ["there", "user"]:
            return "Hello! How can I help you analyze mainframe jobs today?"
        return f"Hello {name}! How can I help you analyze mainframe jobs today?"

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
        
        # Format as readable text instead of JSON
        response = f"Job: {job_name}\n"
        response += f"Predecessors: {', '.join(predecessors) if predecessors else 'None'}\n"
        response += f"Successors: {', '.join(successors) if successors else 'None'}\n"
        if file_triggers:
            response += f"File Triggers: {', '.join(file_triggers)}\n"
        response += f"Total Predecessors: {len(predecessors)}\n"
        response += f"Total Successors: {len(successors)}"
        
        return response                                                                                         
    
    def get_job_criticality(self, job_name: str) -> str:
        """Get the criticality status of a job"""
        try:
            # Clean the input
            job_name = job_name.strip('"').strip("'")
            
            query = """
            SELECT DISTINCT job_name, is_critical 
            FROM mainframe_deps
            WHERE job_name = ? 
            LIMIT 1
            """
            
            result = self.db_conn.execute(query, [job_name]).fetchdf()
            print(f"DEBUG - Query result for {job_name}: {result}")
            
            if result.empty:
                return f"No job found with name '{job_name}'"
            
            is_critical = result['is_critical'].iloc[0]
            return f"Job '{job_name}' is {'a critical' if is_critical == 'Y' else 'not a critical'} job (is_critical = '{is_critical}')"
            
        except Exception as e:
            print(f"Error in get_job_criticality: {str(e)}")
            return f"Error checking criticality for job '{job_name}': {str(e)}"

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


    def show_dependency_graph(self, job_name:str):
        """
        Display dependency graph in Streamlit if it exists
        Args:
            job_name: Name of the job
        Returns:
            bool: True if graph was displayed, False otherwise
        """
        try:
            graph_path = f"graphs/{job_name}_dependencies.png"
            if os.path.exists(graph_path):
                # Use st directly since we're in Streamlit context
                st.image(graph_path, caption=f"Dependency graph for {job_name}")
                return True
            return False
        except Exception as e:
            print(f"Error displaying graph: {e}")
            return False


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
            description="Check if a job is marked as critical (Y) or non-critical (N). Input should be a job name (e.g., 'JOB0004')."
        ),
        Tool(
            name="search_knowledge_articles",
            func=job_analyzer.search_knowledge_articles,
            description="Search knowledge articles for information related to job failures and resolutions any relavent Business impact as well as SLA. Input should be a search query."
        ),
        Tool(
            name="generate_dependency_graph",
            func=job_analyzer.generate_dependency_graph,
            description="Generate a GraphViz visualization of the job's dependencies. Input should be a job name."
        ),
        Tool(
            name="greet_user",
            func=job_analyzer.greet_user,
            description="Greet the user with a friendly message. Input should be the user's name, if available."
        ),
        Tool(
            name="show_dependency_graph",
            func=job_analyzer.show_dependency_graph,
            description="Once the GraphViz vizualization of the job dependencies is generated, this function should be called to display the image to the user.Input should be a job name."
        )
    ]

    # Create prompt with ReAct framework
    system_prompt_text ="""You are an AI assistant specialized in analyzing mainframe jobs.

    You MUST STRICTLY follow these rules:
    1. If the user ONLY says hello, hi, hey, or gives a greeting without asking about jobs, ALWAYS use the greet_user tool FIRST.
    2. For job analysis questions, follow this EXACT sequence:

    You MUST follow this EXACT format for EVERY response:

    Thought: <describe your reasoning>
    Action: <one tool from {tool_names}>
    Action Input: "<input in quotes>"
    Observation: <wait for tool response>
    Thought: <next step or proceed to final answer>
    Final Answer: <your conclusion>

    NEVER skip the Action and Action Input steps.
    NEVER make up information.
    ALWAYS wait for tool responses.
    ONLY use the tools provided.

    Available tools:
    {tools}

    Critical Rules:
    1. For greetings (hello/hi), use ONLY the greet_user tool
    2. For job queries, ALWAYS check criticality first
    3. NEVER skip the Action/Action Input steps
    4. NEVER make up information
    5. ALWAYS wait for tool responses
    6. ONLY use listed tools

    Example 1 - Greeting:
    User: Hello
    Thought: The user is greeting me. I should respond with a greeting.
    Action: the action to take should be from one of the [{tool_names}]
    Action Input: If user provides a name, use that as input.
    Observation: Hello! How can I help you analyze mainframe jobs today?
    Final Answer: Hello! How can I help you analyze mainframe jobs today?

    Example - Job Query:
    User: Tell me about JOB0004
    Thought: I should check if this is a critical job
    Action: get_job_criticality
    Action Input: "JOB0004"
    Observation: Job 'JOB0004' is not a critical job (is_critical = 'N')
    Final Answer: Based on the analysis, JOB0004 is not marked as a critical job.
    
    Begin!
    Quesiton = {input}
    Thought: {agent_scratchpad}
    """

    prompt = ChatPromptTemplate.from_messages([ 
        system_prompt_text,
        # MessagesPlaceholder(variable_name="agent_scratchpad"),
        HumanMessage(content="{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    template = """
    Answer the following questions the best you can using {tools} from {tool_names}
    Begin!
    Quesiton = {input}
    Thought: {agent_scratchpad}
    """
    # react_prompt = PromptTemplate.from_template(template)
    react_prompt = PromptTemplate.from_template(system_prompt_text)

    # ("user", "{input}"),
    # ("assistant", "{agent_scratchpad}")
    # Optional: Add placeholder for chat history if you plan to use memory
    
    agent = create_react_agent(
        llm=job_analyzer.llm, 
        tools=tools, 
        prompt=react_prompt,
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="force",  # Add this
        return_intermediate_steps=True,  # Add this
        max_output_toekns=512,
        agent_error_behavior="retry_with_refinement"  # Add error handling behavior
    )
    
    return agent_executor

# Streamlit UI
def create_streamlit_app():
    st.title("Job Analysis Assistant")
    st.write("Ask questions about job dependencies, criticality, and troubleshooting.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        gemini = st.checkbox(label="Use Gemini")
        if gemini:
            ollama_true = False
        else:
            ollama_true = True
        
        model = st.selectbox(
            "Select Ollama Model", 
            ["llama3", "mistral", "llama2", "llama3:8b", "codellama", "phi3", "deepseek-r1:1.5b"], 
            index=0
        )
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    
    # Initialize analyzer with selected model
    job_analyzer = JobAnalyzer(ollama_model=model, ollama_host=ollama_url,ollama_true=ollama_true)
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