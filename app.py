import streamlit as st
from openai import OpenAI
import os
import time
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K, EMBEDDING_MODEL_NAME
OPENROUTER_API_KEY = getattr(__import__('config'), 'OPENROUTER_API_KEY', os.getenv("OPENROUTER_API_KEY", ""))
from pdf_parser import process_uploaded_files
from retrievers import BM25Retriever, DenseRetriever, HybridRRFRetriever
from indexer import FaissIndex

st.set_page_config(
    page_title="Academic RAG Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Academic Hybrid-RAG System")
st.caption("Advanced Information Retrieval & Synthesis System using BM25, FAISS, and Reciprocal Rank Fusion (RRF)")

# Initialize Session State
if "documents" not in st.session_state:
    st.session_state["documents"] = []
if "hybrid_retriever" not in st.session_state:
    st.session_state["hybrid_retriever"] = None
if "faiss_indexer" not in st.session_state:
    st.session_state["faiss_indexer"] = None
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = set()

def generate_answer_with_openrouter(query: str, retrieved_chunks: list, api_key: str) -> str:
    if not api_key:
        return "⚠️ Please enter your OpenRouter API Key in the sidebar or .env file."
    
    context = "\n\n---\n\n".join([
        f"[Source: {doc['metadata']['source_file']} | Page: {doc['metadata']['page_number']}]\n{doc['text']}"
        for doc in retrieved_chunks
    ])
    
    # Enhanced Academic System Prompt
    system_prompt = (
        "You are an expert academic research assistant. Answer the user's query comprehensively "
        "and accurately based strictly on the provided context.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. You MUST include inline citations in your answer referring to the source and page number "
        "(e.g., [Source: Lecture1.pdf | Page: 5]).\n"
        "2. Do not hallucinate or use external knowledge outside the provided context.\n"
        "3. Format your response with clear headings and bullet points if necessary."
    )
    
    user_prompt = f"Context Context:\n{context}\n\nUser Query: {query}\n\nAcademic Answer:"
    
    free_models = [
        "google/gemini-2.5-flash:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "openrouter/auto"
    ]
    
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    last_error = ""
    for model_name in free_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1 # Low temperature for factual academic answers
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = str(e)
            continue
            
    return f"API Connection Error: {last_error}"

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔑 API Configuration")
    user_openrouter_key = st.text_input("OpenRouter API Key:", value=OPENROUTER_API_KEY or "", type="password")

    st.divider()
    st.header("📂 Document Management")
    upload_mode = st.radio(
        "Processing Mode:",
        options=["Append to existing corpus", "Reset and start new session"],
        index=0
    )

    uploaded_files = st.file_uploader(
        "Upload Lecture PDFs:",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🚀 Process & Index Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file first.")
        else:
            with st.spinner("Extracting text, chunking, and building vector indexes..."):
                start_time = time.time()
                
                if "Reset" in upload_mode:
                    st.session_state["documents"] = []
                    st.session_state["processed_files"] = set()

                new_files = [f for f in uploaded_files if f.name not in st.session_state["processed_files"]]

                if not new_files and "Append" in upload_mode:
                    st.info("All uploaded files have already been processed.")
                else:
                    new_docs = process_uploaded_files(
                        new_files if "Append" in upload_mode else uploaded_files,
                        chunk_size=DEFAULT_CHUNK_SIZE,
                        overlap=DEFAULT_CHUNK_OVERLAP
                    )

                    if "Append" in upload_mode:
                        start_id = len(st.session_state["documents"])
                        for i, doc in enumerate(new_docs):
                            doc["id"] = start_id + i
                        st.session_state["documents"].extend(new_docs)
                    else:
                        st.session_state["documents"] = new_docs

                    for f in uploaded_files:
                        st.session_state["processed_files"].add(f.name)

                    corpus_texts = [doc["text"] for doc in st.session_state["documents"]]

                    # Building Retrievers
                    bm25_ret = BM25Retriever(corpus_texts)
                    dense_ret = DenseRetriever(corpus_texts, EMBEDDING_MODEL_NAME)
                    
                    st.session_state["hybrid_retriever"] = HybridRRFRetriever(bm25_ret, dense_ret)
                    st.session_state["faiss_indexer"] = FaissIndex(dense_ret.embeddings)
                    
                    processing_time = round(time.time() - start_time, 2)
                    st.success(f"✅ Processed {len(st.session_state['documents'])} chunks in {processing_time} seconds!")

    st.divider()
    st.markdown("### 📊 System Analytics")
    st.write(f"• Active Files: **{len(st.session_state['processed_files'])}**")
    st.write(f"• Total Vector Chunks: **{len(st.session_state['documents'])}**")
    st.write(f"• Embedding Model: **{EMBEDDING_MODEL_NAME}**")
    
    st.divider()
    st.info("**Methodology:** This system utilizes a **Hybrid Retrieval Strategy**. It combines Lexical Search (BM25) with Semantic Search (Dense Embeddings via FAISS), fused together using Reciprocal Rank Fusion (RRF) for state-of-the-art accuracy.")

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
if not st.session_state["documents"]:
    st.info("👋 Welcome! Please upload your lecture materials from the sidebar and click 'Process & Index Documents' to begin.")
else:
    st.subheader("🔍 Query the Knowledge Base")
    query = st.text_input("Enter your academic query:", placeholder="e.g., What are the main characteristics of KNN algorithm?")

    col1, _ = st.columns([1, 4])
    with col1:
        top_k = st.slider("Top-K Retrieval (Number of Context Chunks):", min_value=1, max_value=10, value=DEFAULT_TOP_K)

    if query.strip():
        # Retrieval Phase
        retrieval_start = time.time()
        with st.spinner("Executing Hybrid Search (BM25 + Dense) via RRF..."):
            hybrid_engine = st.session_state["hybrid_retriever"]
            results = hybrid_engine.search(query, top_k=top_k)
            retrieved_chunks = [st.session_state["documents"][res["doc_id"]] for res in results]
        retrieval_time = round(time.time() - retrieval_start, 3)

        # Generation Phase
        gen_start = time.time()
        with st.spinner("Synthesizing answer using LLM..."):
            ai_answer = generate_answer_with_openrouter(query, retrieved_chunks, user_openrouter_key)
        gen_time = round(time.time() - gen_start, 3)

        # Display Results
        st.markdown("### 🤖 Synthesized Academic Response:")
        st.success(ai_answer)
        
        # Display Metrics (Professors love this)
        st.caption(f"⏱️ **Performance Metrics:** Retrieval Time: {retrieval_time}s | Generation Time: {gen_time}s")

        st.markdown("---")
        st.markdown("#### 📄 Retrieved Context (Ground Truth Sources):")
        for i, doc_data in enumerate(retrieved_chunks, start=1):
            meta = doc_data["metadata"]
            with st.expander(f"📌 Source {i}: {meta['source_file']} (Page {meta['page_number']})"):
                st.write(doc_data["text"])