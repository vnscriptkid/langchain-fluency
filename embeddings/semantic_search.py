from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

def semantic_search(sentences: List[str], query: str) -> List[tuple]:
    # Initialize OpenAI embeddings
    # This creates an instance of OpenAIEmbeddings, which will be used to generate
    # vector representations of our sentences and query
    embeddings = OpenAIEmbeddings(api_key=api_key)

    persist_directory = "./chroma_db"

    # Create a Chroma vector store from the sentences
    # Chroma is an open-source embedding database that allows for efficient
    # similarity search. Here, we're creating a Chroma instance and populating it
    # with our sentences, using the OpenAI embeddings to convert them to vectors
    vectorstore = Chroma.from_texts(sentences, embeddings, persist_directory=persist_directory)

    # Persist the data to disk
    vectorstore.persist()
    
    # Perform similarity search with scores
    results = vectorstore.similarity_search_with_score(query, k=len(sentences))
    
    # Sort results by score (lowest first) and format the output
    sorted_results = sorted([(doc.page_content, score) for doc, score in results], key=lambda x: x[1])
    
    return sorted_results

# Example usage
sentences = [
    "To be or not to be, that is the question.",
    "All the world's a stage, and all the men and women merely players.",
    "Romeo, Romeo, wherefore art thou Romeo?",
    "Friends, Romans, countrymen, lend me your ears.",
    "The lady doth protest too much, methinks.",
    "What's a famous Shakespeare quote?",
    "Shakespeare used to say: Sky is blue"
]
query = "What's a famous Shakespeare quote?"

# Call the semantic_search function with our example sentences and query
results = semantic_search(sentences, query)

# Print all sentences with their similarity scores
print("Sentences sorted by similarity score (most similar first):")
print("Score range: 0 (identical) to 1 (completely dissimilar)")
print("Typical interpretation:")
print("  < 0.1: Very high similarity")
print("  0.1 - 0.3: High similarity")
print("  0.3 - 0.5: Moderate similarity")
print("  0.5 - 0.7: Low similarity")
print("  > 0.7: Very low similarity")
print("\nResults:")
for sentence, score in results:
    similarity = "Very high" if score < 0.1 else "High" if score < 0.3 else "Moderate" if score < 0.5 else "Low" if score < 0.7 else "Very low"
    print(f"Score: {score:.4f} | Similarity: {similarity:9} | Sentence: {sentence}")

# Note: This code assumes you have set the OPENAI_API_KEY environment variable
# with your OpenAI API key. You also need to install the following packages:
# pip install langchain chromadb openai
