
import hashlib
import os
from pathlib import Path

import streamlit as st


# -------------------- CONFIGURATION --------------------

ROOT = Path(__file__).resolve().parent
WIKI_DB = ROOT / "db"
UPLOAD_DB = ROOT / "uploaded_db"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

WIKI_NAME = "wikipedia"
UPLOAD_NAME = "uploaded_documents"

CHUNK_WORDS = 175
CHUNK_OVERLAP = 40
TOP_K = 3


# -------------------- STREAMLIT PAGE --------------------

st.set_page_config(
    page_title="Grounded RAG",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1020;
        color: #e8edf8;
    }

    [data-testid="stSidebar"] {
        background: #10172a;
    }

    .hero {
        padding: 1.5rem;
        border: 1px solid #283653;
        border-radius: 18px;
        background: linear-gradient(135deg, #172440, #111827);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.4rem;
        color: #f3f6ff;
    }

    .hero p {
        color: #b8c5dd;
        margin-bottom: 0;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #26334d;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1> Grounded RAG Assistant</h1>
        <p>
            Ask questions about your documents or the indexed Wikipedia
            collection. Answers are generated using retrieved context.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- CACHED RESOURCES --------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    # Import only when the model is actually needed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_wiki_client():
    if not WIKI_DB.exists():
        return None

    import chromadb

    return chromadb.PersistentClient(path=str(WIKI_DB))


@st.cache_resource
def get_upload_client():
    import chromadb

    UPLOAD_DB.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=str(UPLOAD_DB))


@st.cache_resource
def get_wiki_collection():
    client = get_wiki_client()

    if client is None:
        return None

    try:
        return client.get_collection(name=WIKI_NAME)
    except Exception:
        return None


@st.cache_resource
def get_upload_collection():
    client = get_upload_client()

    try:
        return client.get_collection(name=UPLOAD_NAME)
    except Exception:
        return None


def get_api_key():
    key = os.getenv("GROQ_API_KEY")

    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            key = None

    return key


# -------------------- DOCUMENT PROCESSING --------------------

def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="replace",
        )

    if filename.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(uploaded_file)
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    raise ValueError("Only PDF and TXT files are supported.")


def chunk_text(
    text,
    chunk_words=CHUNK_WORDS,
    overlap=CHUNK_OVERLAP,
):
    words = text.split()

    if not words:
        return []

    if overlap >= chunk_words:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    chunks = []
    step = chunk_words - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words])

        if chunk.strip():
            chunks.append(chunk)

        if start + chunk_words >= len(words):
            break

    return chunks


def index_uploaded_file(uploaded_file):
    text = extract_text(uploaded_file)

    if not text.strip():
        raise ValueError(
            "No extractable text found. Scanned PDFs may require OCR."
        )

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No usable text chunks were created.")

    model = load_embedding_model()

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    client = get_upload_client()

    collection = client.get_or_create_collection(
        name=UPLOAD_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    file_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()[:16]

    ids = [
        f"{file_hash}_{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "filename": uploaded_file.name,
            "chunk_index": index,
            "file_hash": file_hash,
        }
        for index in range(len(chunks))
    ]

    # Replace an earlier indexing of the same file.
    collection.delete(where={"file_hash": file_hash})

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Clear cached collection so the next query sees updated data.
    get_upload_collection.clear()

    return len(chunks)


# -------------------- RETRIEVAL --------------------

def retrieve_documents(question, collection, top_k=TOP_K, method="advanced"):
    from src.advanced_retrieval import advanced_retrieve

    return advanced_retrieve(
        question=question,
        collection=collection,
        top_k=top_k,
        method=method,
    )


# -------------------- GROQ ANSWER GENERATION --------------------

@st.cache_resource
def get_groq_client(api_key):
    from groq import Groq

    return Groq(api_key=api_key)


def generate_answer(question, sources):
    if not sources:
        return (
            "I couldn't find relevant information in the selected "
            "knowledge base. Try another question or upload a document "
            "containing the answer."
        )

    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Set it in your terminal "
            "before starting Streamlit."
        )

    context_parts = []

    for index, source in enumerate(sources, start=1):
        filename = source["metadata"].get(
            "filename",
            "Wikipedia collection",
        )

        context_parts.append(
            f"[Source {index}: {filename}]\n{source['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful Retrieval-Augmented Generation assistant.

Answer the user's question using ONLY the retrieved context below.

Rules:
1. Do not invent unsupported facts.
2. If the context does not contain the answer, say so.
3. Explain in simple, direct language.
4. Cite supporting passages using [Source 1], [Source 2], etc.
5. Distinguish stated facts from reasonable inferences.
6. Do not claim to have read anything outside the context.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    client = get_groq_client(api_key)
    last_error = None

    for model_name in LLM_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.2,
                max_completion_tokens=1024,
            )

            answer = response.choices[0].message.content

            if answer and answer.strip():
                return answer.strip()

            last_error = RuntimeError(
                f"{model_name} returned an empty answer."
            )

        except Exception as error:
            last_error = error
            print(f"Groq model {model_name} failed: {error}")

    raise RuntimeError(
        f"Answer generation failed. Last error: {last_error}"
    ) from last_error


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.title("⚙️ Settings")

    knowledge_source = st.radio(
        "Knowledge base",
        [
            "Uploaded documents",
            "Wikipedia collection",
        ],
    )

    retrieval_method = st.selectbox(
        "Retrieval strategy",
        options=["advanced", "hybrid", "dense", "bm25"],
        format_func=lambda value: {
            "advanced": "Hybrid + Cross-Encoder (Recommended)",
            "hybrid": "Hybrid + RRF",
            "dense": "Dense semantic search",
            "bm25": "BM25 keyword search",
        }[value],
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=1,
        max_value=6,
        value=TOP_K,
    )

    st.divider()

    st.subheader("📄 Upload documents")

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button(
        "Index uploaded documents",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning("Upload at least one PDF or TXT file first.")
        else:
            total_chunks = 0
            progress = st.progress(0)

            for index, file in enumerate(uploaded_files):
                try:
                    with st.spinner(f"Indexing {file.name}..."):
                        count = index_uploaded_file(file)

                    total_chunks += count

                    st.success(
                        f"{file.name}: indexed {count} chunks."
                    )

                except Exception as error:
                    st.error(f"{file.name}: {error}")

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            st.info(
                f"Indexing finished. {total_chunks} chunks processed."
            )

    st.divider()

    st.caption("Answer generation: Groq API")
    st.caption("Embeddings: Sentence Transformers")
    st.caption("Vector database: ChromaDB")


# -------------------- CHAT INTERFACE --------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for index, source in enumerate(
                    message["sources"],
                    start=1,
                ):
                    filename = source["metadata"].get(
                        "filename",
                        "Wikipedia collection",
                    )

                    st.markdown(
                        f"**Source {index}: {filename}**"
                    )

                    st.write(source["text"])

                    if source.get("distance") is not None:
                        st.caption(
                            "Cosine distance: "
                            f"{source['distance']:.4f}"
                        )


question = st.chat_input(
    "Ask a question about your knowledge base..."
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your knowledge base..."):
                if knowledge_source == "Uploaded documents":
                    collection = get_upload_collection()
                else:
                    collection = get_wiki_collection()

                sources = retrieve_documents(
                    question,
                    collection,
                    top_k=top_k,
                    method=retrieval_method,
                )

            with st.spinner("Generating an answer with Groq..."):
                answer = generate_answer(question, sources)

            st.markdown(answer)

            if sources:
                with st.expander("View retrieved sources"):
                    for index, source in enumerate(
                        sources,
                        start=1,
                    ):
                        filename = source["metadata"].get(
                            "filename",
                            "Wikipedia collection",
                        )

                        st.markdown(
                            f"**Source {index}: {filename}**"
                        )

                        st.write(source["text"])

                        if source.get("distance") is not None:
                            st.caption(
                                "Cosine distance: "
                                f"{source['distance']:.4f}"
                            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as error:
            error_message = f"Something went wrong: {error}"
            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "RAG retrieves relevant passages before generating answers. "
    "Always verify important information against the original sources."
)
