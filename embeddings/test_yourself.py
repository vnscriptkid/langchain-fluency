import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
# This allows us to securely store and access sensitive information like API keys
load_dotenv()

# Get the OpenAI API key from environment variables
# This is more secure than hardcoding the API key in the script
api_key = os.getenv("OPENAI_API_KEY")

def load_and_split_document(file_path):
    # Load the document using TextLoader
    # TextLoader is used to load text from a file
    loader = TextLoader(file_path)
    document = loader.load()

    # Split the document into chunks using RecursiveCharacterTextSplitter
    # This helps in processing large documents by breaking them into smaller, manageable pieces
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Each chunk will be approximately 1000 characters long
        chunk_overlap=0,  # No overlap between chunks
        length_function=len  # Use the built-in len function to measure text length
    )
    chunks = text_splitter.split_documents(document)
    return chunks

def create_vectorstore(chunks):
    # Create embeddings using OpenAIEmbeddings
    # Embeddings are vector representations of text that capture semantic meaning
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # Create a vector store using Chroma
    # This stores the document chunks as vectors for efficient similarity search
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

def setup_qa_chain(vectorstore):
    # Create a retrieval-based QA chain
    # This chain will use the vector store to find relevant information and generate answers
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(api_key=api_key),  # Use ChatOpenAI as the language model
        chain_type="stuff",  # "stuff" chain type passes all retrieved documents to the model at once
        retriever=vectorstore.as_retriever()  # Use the vector store for retrieval
    )
    return qa_chain

def main():
    # Load and split the document
    chunks = load_and_split_document("document.txt")

    # Create vector store from the document chunks
    vectorstore = create_vectorstore(chunks)

    # Set up QA chain using the vector store
    qa_chain = setup_qa_chain(vectorstore)

    while True:
        # Get user question
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        # Get answer using the QA chain
        answer = qa_chain.invoke(question)
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
