# Grounded RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from Wikipedia data or user-uploaded documents. It combines semantic search with a large language model to provide answers grounded in relevant context.

## Features

- **Question answering:** Ask questions about indexed knowledge and uploaded documents.
- **Wikipedia retrieval:** Search a pre-ingested Wikipedia dataset.
- **Document upload:** Upload PDF and TXT files to query their contents.
- **Semantic search:** Generate text embeddings using Sentence Transformers.
- **Vector database:** Store and retrieve document chunks using ChromaDB.
- **LLM integration:** Generate responses using models available through the Groq API.
- **Web interface:** Use Streamlit to interact with the assistant through a browser.
- **Terminal interface:** Run the command-line question-answering module directly from the terminal.

## Tech Stack

- Python
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers — `all-MiniLM-L6-v2`
- ChromaDB
- Groq API
- Streamlit
- PyPDF
- Hugging Face Datasets

## How It Works

1. **Document ingestion:** Wikipedia articles or uploaded files are processed into smaller text chunks.
2. **Embedding generation:** Sentence Transformers converts the chunks into numerical vector representations.
3. **Vector storage:** ChromaDB stores the vectors and their associated text.
4. **Semantic retrieval:** A user's question is used to retrieve relevant passages from the knowledge base.
5. **Answer generation:** The retrieved context is passed to a Groq-hosted language model to generate an answer.
6. **Response display:** The answer is returned through the terminal or the Streamlit interface, depending on how the application is launched.

## Project Structure

```text
rag-experiment-main/
├── app.py                 # Streamlit web application
├── src/
│   ├── __init__.py
│   ├── ingest.py           # Document ingestion
│   ├── query.py            # Question-answering module
│   ├── evaluate_rag.py     # RAG evaluation
│   └── plot_results.py     # Results visualization
├── db/                     # Persistent Wikipedia vector database
├── uploaded_db/             # Uploaded-document vector database
├── requirements.txt
└── README.md
```

*Note: The exact files may differ depending on the current repository version. The `uploaded_db/` directory may be created automatically when documents are ingested.*

## Setup

### 1. Clone or open the repository

Open PowerShell in the project directory:

```powershell
cd C:\Users\acer\Videos\rag-experiment-main
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If the virtual environment does not exist, create it:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Groq API key

Set your API key in the current PowerShell session:

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

Replace the placeholder with your own API key. Do not commit API keys to GitHub.

## Run the Application

### Option 1: Terminal question answering

```powershell
python -m src.query
```

Use this mode to ask questions directly from the terminal, provided `src/query.py` implements a runnable command-line interface.

### Option 2: Streamlit web interface

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually:

`http://localhost:8501`

## Evaluation

The project includes an evaluation module and a results-visualization module for assessing and inspecting RAG performance.

Run these modules only with the arguments and configuration supported by their current implementations:

```powershell
python -m src.evaluate_rag
python -m src.plot_results
```

## Data and Storage

- `db/` stores the persistent Wikipedia vector database, if it has already been created.
- `uploaded_db/` is used for indexed user-uploaded documents.
- Vector databases may need to be regenerated if their source data, embedding model, or collection configuration changes.

## Limitations

- Answer quality depends on the relevance and completeness of the retrieved passages.
- The language model may still produce inaccurate information.
- API access requires a valid Groq API key.
- The terminal and Streamlit entry points depend on the functionality implemented in the corresponding Python files.

## Resume Summary

Developed a RAG-based question-answering assistant using Python, Sentence Transformers, ChromaDB, Groq API, and Streamlit. Implemented semantic retrieval over Wikipedia data and uploaded documents, with terminal and browser-based interaction.

## Future Improvements

- Add a fixed question-and-answer evaluation dataset.
- Measure retrieval quality and answer faithfulness.
- Add automated tests for ingestion, retrieval, and response generation.
- Improve source citations and document-level filtering.
