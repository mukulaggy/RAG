# app.py
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings

from pymilvus import MilvusClient
from groq import Groq
from tqdm import tqdm
import sys
from pymilvus import connections, utility, Collection
from dotenv import load_dotenv

load_dotenv()

# ======= LangChain Tracing & API Key =======
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Custom print function that outputs to both terminal and Streamlit
def log_message(message, streamlit_output=True):
    print(message)  # This will show in terminal
    if streamlit_output:
        st.write(f"`{message}`")  # This will show in Streamlit app

# Function to get API key from either secrets or environment variables
def get_api_key(key_name, secrets_key=None):
    # Try to get from Streamlit secrets first
    try:
        if secrets_key and hasattr(st, 'secrets') and st.secrets.get(secrets_key):
            return st.secrets[secrets_key]
    except:
        pass
    
    # Try to get from environment variables
    env_value = os.getenv(key_name)
    if env_value:
        return env_value
    
    # Try alternative environment variable names
    alt_env_value = os.getenv(f"GROQ_{key_name}") or os.getenv(f"{key_name}_KEY")
    if alt_env_value:
        return alt_env_value
    
    return None

# ======= Streamlit UI =======
st.title("RAG Demo with Nomic Embeddings & Groq LLM")

# Display API key status
groq_api_key = get_api_key("GROQ_API_KEY", "GROQ_API_KEY")
if groq_api_key:
    log_message("✓ GROQ_API_KEY found", streamlit_output=False)
else:
    log_message("✗ GROQ_API_KEY not found in secrets or environment variables")

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
web_url = st.text_input("Or enter a webpage URL:")

if st.button("Ingest Documents"):
    docs = []
    
    # ---------- PDF Loader ----------
    if uploaded_file:
        try:
            log_message("Processing PDF file...")
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_file_path)
            docs += loader.load()
            os.unlink(tmp_file_path)  # Clean up temp file
            log_message(f"Loaded {len(docs)} pages from PDF")
        except Exception as e:
            error_msg = f"Error loading PDF: {str(e)}"
            log_message(error_msg)
            st.error(error_msg)

    # ---------- Web Loader ----------
    if web_url:
        try:
            log_message(f"Loading web page: {web_url}")
            loader = WebBaseLoader(web_url)
            web_docs = loader.load()
            docs += web_docs
            log_message(f"Loaded {len(web_docs)} documents from web page")
        except Exception as e:
            error_msg = f"Error loading web page: {str(e)}"
            log_message(error_msg)
            st.error(error_msg)

    if not docs:
        warning_msg = "No documents loaded. Please upload a PDF or enter a URL."
        log_message(warning_msg)
        st.warning(warning_msg)
        st.stop()

    # ---------- Chunk Documents ----------
    try:
        log_message("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(docs)
        log_message(f"Created {len(docs)} document chunks.")
    except Exception as e:
        error_msg = f"Error splitting documents: {str(e)}"
        log_message(error_msg)
        st.error(error_msg)
        st.stop()

    # ---------- Nomic Embeddings ----------
    try:
        log_message("Initializing embeddings...")
        class NomicEmbeddings(Embeddings):
            def __init__(self):
                # Initialize HuggingFace model once
                self.hf_embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                )

            def embed_documents(self, texts):
                embeddings_list = []
                log_message(f"Generating embeddings for {len(texts)} documents...")
                
                for i, t in enumerate(tqdm(texts, desc="Generating embeddings", file=sys.stdout)):
                    log_message(f"Embedding document {i+1}/{len(texts)}: {t[:50]}...", streamlit_output=False)
                    embedding = self.hf_embeddings.embed_query(t)
                    log_message(f"Generated embedding with {len(embedding)} dimensions", streamlit_output=False)
                    embeddings_list.append(embedding)
                
                return embeddings_list

            def embed_query(self, text):
                log_message(f"Generating query embedding for: {text[:50]}...")
                embedding = self.hf_embeddings.embed_query(text)
                log_message(f"Query embedding dimensions: {len(embedding)}")
                return embedding

        embeddings = NomicEmbeddings()
        log_message("Embeddings initialized successfully!")
    except Exception as e:
        error_msg = f"Error initializing embeddings: {str(e)}"
        log_message(error_msg)
        st.error(error_msg)
        st.error("Make sure Ollama is running and 'nomic-embed-text' model is available")
        st.stop()

    # ---------- Milvus Vector Store ----------
    try:
        log_message("Storing documents in vector database...")
        milvus_client = MilvusClient(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("ZILLIZ_API_KEY"),
            secure=True
        )
        collection_name = "rag_demo_nomic"
        
        log_message("Creating embeddings and storing in Milvus...")
        vectorstore = Milvus.from_documents(
            docs,
            embeddings,
            connection_args={
                "uri": os.getenv("MILVUS_URI"),
                "token": os.getenv("ZILLIZ_API_KEY"),
                "secure": True,
                "db_name": "default"
            },
            collection_name=collection_name
        )
        st.session_state.vectorstore = vectorstore
        log_message("Documents successfully ingested into vector database!")
        st.success("Documents successfully ingested into vector database!")
    except Exception as e:
        error_msg = f"Error connecting to Milvus: {str(e)}"
        log_message(error_msg)
        st.error(error_msg)
        st.error("Make sure Milvus is running on localhost:19530")
        st.stop()

    # ---------- Groq LLM Wrapper ----------
    try:
        log_message("Initializing Groq LLM...")
        
        # Get API key using our function
        groq_api_key = get_api_key("GROQ_API_KEY", "GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in secrets or environment variables")
        
        client = Groq(api_key=groq_api_key)
        
        class GroqLLM:
            """Minimal LangChain-compatible LLM wrapper"""
            def __init__(self, client, model="llama-3.3-70b-versatile"):
                self.client = client
                self.model = model
                log_message(f"Initialized LLM with model: {model}")

            def __call__(self, prompt):
                log_message(f"Sending prompt to LLM: {prompt[:100]}...")
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300,
                )
                response_text = resp.choices[0].message.content
                log_message(f"Received LLM response: {response_text[:100]}...")
                return response_text

        st.session_state.llm = GroqLLM(client)
        log_message("LLM initialized successfully!")
        st.success("LLM initialized successfully!")
    except Exception as e:
        error_msg = f"Error initializing LLM: {str(e)}"
        log_message(error_msg)
        st.error(error_msg)
        st.error("Make sure GROQ_API_KEY is set in Streamlit secrets or environment variables")
        
        # Provide instructions for setting up the API key
        st.info("""
        **How to set up GROQ_API_KEY:**
        
        1. **Environment Variable (Recommended):**
           ```bash
           export GROQ_API_KEY="your-api-key-here"
           ```
           
        2. **Streamlit Secrets:**
           Create a `.streamlit/secrets.toml` file with:
           ```toml
           GROQ_API_KEY = "your-api-key-here"
           ```
        """)
        st.stop()

# Question answering section
st.divider()
question = st.text_input("Ask a question about your documents:")

if st.button("Get Answer") and question:
    if not st.session_state.vectorstore or not st.session_state.llm:
        warning_msg = "Please ingest documents first before asking questions."
        log_message(warning_msg)
        st.warning(warning_msg)
        st.stop()
    
    try:
        # ---------- Prompt Template ----------
        prompt_template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you don't know.
Context:
{context}

Question:
{question}

Answer concisely and clearly.
"""
        PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        # ---------- RetrievalQA Chain ----------
        class CustomRAGChain:
            def __init__(self, retriever, llm, prompt_template):
                self.retriever = retriever
                self.llm = llm
                self.prompt_template = prompt_template
                log_message("RAG chain initialized")

            def run(self, question):
                log_message(f"Searching for relevant documents for query: {question}")
                docs = self.retriever.get_relevant_documents(question)
                log_message(f"Found {len(docs)} relevant documents")
                
                for i, doc in enumerate(docs):
                    log_message(f"Document {i+1}: {doc.page_content[:100]}...", streamlit_output=False)
                
                context = "\n".join([d.page_content for d in docs])
                prompt = self.prompt_template.format(context=context, question=question)
                log_message(f"Final prompt length: {len(prompt)} characters")
                
                return self.llm(prompt)

        log_message("Setting up retriever...")
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = CustomRAGChain(retriever, st.session_state.llm, PROMPT)
        
        log_message("Generating answer...")
        answer = qa_chain.run(question)

        st.subheader("Answer:")
        st.write(answer)
        log_message(f"Final answer: {answer}")
        
        # Show context sources
        with st.expander("See relevant context used"):
            retrieved_docs = retriever.get_relevant_documents(question)
            for i, doc in enumerate(retrieved_docs):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content)
                st.divider()
                
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        log_message(error_msg)
        st.error(error_msg)

        