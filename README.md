# Local RAG Pipeline with ChromaDB and TF-IDF

This project demonstrates two implementations of a Retrieval-Augmented Generation (RAG) pipeline using a local Large Language Model (LLM) with Ollama. The goal is to answer questions based on a collection of classic literature texts.

## Features

*   **Two Retrieval Methods**:
    1.  **Vector-Based (app.py)**: Uses `ChromaDB` for vector storage and `Sentence-Transformers` for creating text embeddings. This is a modern, semantic-based approach.
    2.  **TF-IDF-Based (tf_idf_rag.py)**: A classic information retrieval method using `scikit-learn`'s `TfidfVectorizer` and cosine similarity.
*   **Local LLM Integration**: Connects to a local Ollama instance to generate answers, ensuring privacy and no API costs.
*   **Automated Data Fetching**: Includes a script (`download_data.py`) to download public domain books from Project Gutenberg.
*   **Interactive CLI**: Allows you to ask questions to the RAG pipeline directly from your terminal.

## Project Structure

*   **`download_data.py`**: Fetches the text files listed in the script and saves them to the `data/` directory.
*   **`app.py`**: The primary RAG implementation. It loads documents, splits them into chunks, generates embeddings, stores them in ChromaDB, and uses Ollama to answer queries based on retrieved context.
*   **`tf_idf_rag.py`**: An alternative, simpler RAG implementation that uses TF-IDF to find relevant documents. It's useful for comparison.
*   **`pyproject.toml`**: Defines the project's dependencies.

## Setup and Installation

### Prerequisites

1.  **Python 3.11+**
2.  **Ollama**: You must have Ollama installed and running. You can download it from [ollama.com](https://ollama.com/).

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ms-kumar/Local-Rag-ChromaDB-TF-IDF.git
    cd Local-Rag-ChromaDB-TF-IDF
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    The project uses a [pyproject.toml](http://_vscodecontentref_/6) file. Install the dependencies using pip:
    ```bash
    pip install .
    ```
    This command reads the [pyproject.toml](http://_vscodecontentref_/7) file and installs all required packages.

4.  **Download LLM Models:**
    Pull the necessary models for Ollama. This project is configured to use `gemma:2b` for faster responses, but you can easily switch to `llama3` or others in the code.
    ```bash
    ollama pull gemma:2b
    ollama pull llama3
    ```

## How to Run

1.  **Download the Data:**
    First, run the [download_data.py](http://_vscodecontentref_/8) script to fetch the text files.
    ```bash
    python download_data.py
    ```
    This will create a [data](http://_vscodecontentref_/9) directory and populate it with `.txt` files.

2.  **Run the RAG Pipeline:**
    You have two options for the RAG pipeline.

    *   **Option A: Vector-Based RAG (Recommended)**
        Run [app.py](http://_vscodecontentref_/10) to start the interactive RAG pipeline using ChromaDB.
        ```bash
        python app.py
        ```
        The script will first process and "ingest" the documents into the vector database. This might take a minute. Afterward, you can start asking questions.

    *   **Option B: TF-IDF Based RAG**
        Run [tf_idf_rag.py](http://_vscodecontentref_/11) to use the classic TF-IDF approach.
        ```bash
        python tf_idf_rag.py
        ```

3.  **Interact with the Pipeline:**
    Once the application is running, you will be prompted to enter a query.
    ```
    Enter your query: What did Frankenstein's monster desire?
    ```
    Type `exit` to quit the application.
