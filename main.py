import os
from dotenv import load_dotenv
import rag_core
import cli_interface


def main():
#   Load environment variables at the main entry point
    load_dotenv()

    print("--- Welcome to your Modular RAG Application ---")

    pdf_path = cli_interface.get_pdf_path()
    if pdf_path is None:
        print("No valid PDF path provided. Exiting.")
        return

    documents = rag_core.load_documents(pdf_path)
    if documents is None:
        print("Failed to load documents. Exiting.")
        return


    chunks = rag_core.split_documents(documents)

    pdf_filename = os.path.basename(pdf_path)
    vector_store = rag_core.create_vector_store(chunks, pdf_filename)

    qa_chain = rag_core.setup_rag_chain(vector_store)


    if qa_chain is None:
        print("RAG chain could not be initialized. Exiting.")
        return

    cli_interface.run_qa_loop(qa_chain, pdf_filename)


if __name__ == "__main__":
    main()