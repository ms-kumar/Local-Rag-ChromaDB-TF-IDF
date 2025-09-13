__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
# from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import ollama
from sentence_transformers import SentenceTransformer

# Explicitly initialize the ChromaDB client
# This helps avoid potential import issues on some systems.
client = chromadb.Client()

def get_embedding_function(model_name="all-MiniLM-L6-v2"):
    """
    Returns a SentenceTransformer embedding function.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

def load_documents(directory):
    """
    Loads and splits all text files from a specified directory into a list of document chunks.
    """
    documents = []
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist. Please run the download_data.py script first.")
        return documents

    print(f"Loading and chunking documents from '{directory}'...")
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Split text into chunks (e.g., by paragraphs)
                chunks = text.split('\n\n')
                # Filter out empty or very short chunks
                chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 100]
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "source": f"{filename}-chunk-{i}"
                    })

            except Exception as e:
                print(f"Error reading or chunking file {filename}: {e}")
    
    return documents

def setup_chromadb(documents, collection_name="documents_collection"):
    """
    Sets up the ChromaDB collection, clearing it first if it exists,
    and then loading the provided document chunks.
    """
    # Delete the collection if it exists, to ensure a fresh start
    # if collection_name in [c.name for c in client.list_collections()]:
    #     client.delete_collection(name=collection_name)

    # Create a new collection
    collection = client.create_collection(
        name=collection_name,
        embedding_function=get_embedding_function()
    )

    print(f"Loading {len(documents)} document chunks into ChromaDB collection '{collection_name}'...")

    # Extract texts and generate IDs from the document objects
    texts = [doc["text"] for doc in documents]
    ids = [doc["source"] for doc in documents]

    # Add documents to the collection in batches
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        collection.add(
            ids=batch_ids,
            documents=batch_texts
        )
        print(f"Added batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

    print("Documents loaded successfully.")
    return collection

def retrieve_context(collection, query, n_results=5):
    """
    Retrieves the most relevant context from ChromaDB for a given query.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if results and results['documents'] and results['documents'][0]:
        retrieved_docs = results['documents'][0]
        # For simplicity, we'll just use the most relevant document's source.
        # A more advanced implementation might check all retrieved docs.
        source_id = results['ids'][0][0] 
        # Assuming the ID format is "doc_X" and filename is stored elsewhere or derived
        # This part needs a robust way to link ID back to filename if needed.
        # For now, we'll just indicate the source is from the vector store.
        source_info = f"Vector Store (ID: {source_id})"
        return "\n".join(retrieved_docs), source_info
    
    return None, None

def ingest_documents(collection, documents, model_name='all-MiniLM-L6-v2'):
    """
    Embeds documents and ingests them into the ChromaDB collection.
    """
    print(f'Loading embedding model: {model_name}')
    model = SentenceTransformer(model_name)
    print("Embedding documents and ingesting into ChromaDB...")
    for doc in documents:
        # Create an embedding for the document text
        embedding = model.encode(doc['text']).tolist()
        
        # Ingest the document into the collection
        collection.add(
            document = [doc['text']],
            metadata = [{'source' : doc['source']}],
            embeddings = [embedding],
            ids=[doc[spurce]]
        )
        print("Document ingestion complete.")

def query_chromadb(collection, query, n_results=5):
    """
    Queries the ChromaDB collection for the most relevant documents.
    """
    # Create an embedding for the query
    query_embedding = model.encode(query).tolist()
    
    # Query the collection for the most similar documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results

def retrieve_context(collection, query, n_results=5):
    """
    Retrieves the most relevant context from ChromaDB for a given query.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if results and results['documents'] and results['documents'][0]:
        retrieved_docs = results['documents'][0]
        # For simplicity, we'll just use the most relevant document's source.
        # A more advanced implementation might check all retrieved docs.
        source_id = results['ids'][0][0] 
        # Assuming the ID format is "doc_X" and filename is stored elsewhere or derived
        # This part needs a robust way to link ID back to filename if needed.
        # For now, we'll just indicate the source is from the vector store.
        source_info = f"Vector Store (ID: {source_id})"
        return "\n".join(retrieved_docs), source_info
    
    return None, None

def generate_response_with_ollama(query, context, model_name="gemma:2b"):
    """
    Generates a final response using a local Ollama LLM with the provided context.
    """
    # The prompt engineering here is crucial. We instruct the model to use the context.
    prompt = f"""Using the following context, answer the query. If the answer is not in the context, state that you don't know.
    Context: {context}
    Query: {query}
    
    Answer:"""
    print(f"Generating response using local model '{model_name}'...")
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {e}. Please ensure Ollama is running and the model '{model_name}' is downloaded."

def main():
    """
    Main function to run the RAG pipeline.
    """
    # Load documents from the specified directory
    documents = load_documents("data")
    if not documents:
        print("No documents found. Exiting.")
        return
    
    # Setup ChromaDB
    collection = setup_chromadb(documents)

    print("\nStarting RAG pipeline. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        
        # Retrieve relevant context
        retrieved_context, source_info = retrieve_context(collection, query, n_results=5) # Increased n_results to 5

        if retrieved_context:
            print(f"\nRetrieved context from: {source_info}")
            # Generate response using the local model
            final_response = generate_response_with_ollama(query, retrieved_context)
            print("\nGenerated Response:")
            print(final_response)
        else:
            print("No relevant context found.")

if __name__ == "__main__":
    main()
