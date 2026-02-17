import os
from typing import Optional
from langchain_classic.chains.retrieval_qa.base import RetrievalQA



#   --- User Input Functions ---
def get_pdf_path() -> Optional[str]:
    """Prompts the user for a PDF file path and validates it."""
    while True:
        pdf_path = input("Please enter the path to your PDF file (e.g., 'my_document.pdf' or 'path/to/my_document.pdf'): ").strip()
        if os.path.exists(pdf_path) and pdf_path.lower().endswith(".pdf"):
            return pdf_path
        else:
            print(f"Error: '{pdf_path}' is not a valid PDF file or does not exist. Please try gain.")
#   Let the user decide if they want to exit or try again
            retry = input("Do you want to try again? (y/n): ").lower()
            if retry != 'y':
                return None


#   --- Interactive Q&A Loop ---
def run_qa_loop(qa_chain: RetrievalQA, pdf_filename: str):
    """Runs the main question and answer loop for the user."""
    print("\n--- RAG Application Ready ---")
    print(f"Knowledge Base: {pdf_filename}")
    print("Ask questions about the document. Type 'exit' to quit.")


    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            print("Exiting RAG application. Goodbye!")
            break

        if not query.strip():
            print("Please enter a question.")
            continue


    try:
        # 🔹 Correct modern call
        result = qa_chain.invoke({"query": query}) 
        print("\n--- Answer ---")
        print(result["result"]) 
        print("\n--- Sources ---")
        for i, doc in enumerate(result.get("source_documents", [])):
            source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
            print(f"Source {i+1}: Page {doc.metadata.get('page', 'N/A')}, from '{source_name}'")
            
    except Exception as e:
        print(f"An error occurred during retrieval or generation: {e}")
        print("Please check your LLM configuration and internet connection if applicable.")
