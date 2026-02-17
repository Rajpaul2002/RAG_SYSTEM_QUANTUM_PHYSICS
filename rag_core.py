import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA




# LLM (HuggingFace Hub)
from langchain_huggingface import HuggingFaceEndpoint

# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceHub


#   Load environment variables
load_dotenv()


#   --- Configuration for RAG Core ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Switched to a more reliable model on the free tier
LLM_MODEL_ID = "google/flan-t5-base"


#   --- Document Loading ---
def load_documents(pdf_path: str):
    print(f"Loading document from {pdf_path}...")
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            print("Warning: No documents were loaded from the PDF. It might be empty or unreadable.")
            return None
        print(f"Loaded {len(documents)} pages.")
        return documents
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None


#   --- Text Splitting ---
def split_documents(documents):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


#   --- Embedding and Vector Store ---
def create_vector_store(chunks, pdf_filename: str):
    base_name = os.path.splitext(pdf_filename)[0]
    faiss_db_path = f"faiss_index_{base_name}"


    print("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(faiss_db_path):
        print(f"Loading existing FAISS index from {faiss_db_path}")
        vector_store = FAISS.load_local(
            faiss_db_path,
            embeddings,
            allow_dangerous_deserialization=True # Necessary for recent FAISS versions
        )
    else:
        print("Building new FAISS index (this may take a few minutes)...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(faiss_db_path)
        print(f"FAISS index saved to {faiss_db_path}")
    return vector_store

def setup_rag_chain(vector_store):
    print("Setting up RetrievalQA chain...")
    try:
        hf_api_token = os.getenv("HF_API_TOKEN")
        if not hf_api_token:
            raise ValueError("HF_API_TOKEN not found in .env. Please set it.")

        # Modern Hugging Face LLM
        llm = HuggingFaceEndpoint(
            repo_id=LLM_MODEL_ID,
            temperature=0.1,
            max_new_tokens=512,
            huggingfacehub_api_token=hf_api_token
        )

        # --- Test LLM ---
        test_prompt = "Explain blackbody radiation in simple terms."
        print("\n--- Testing LLM ---")
        response = llm.invoke(test_prompt)  # ✅ pass as string
        print("LLM Output:\n", response)

    except Exception as e:
        print(f"Error initializing HuggingFaceEndpoint LLM: {e}")
        return None

    # Setup RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain
