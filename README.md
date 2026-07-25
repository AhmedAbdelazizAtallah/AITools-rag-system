# 🎓 Academic Hybrid-RAG System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![LLM](https://img.shields.io/badge/LLM-GPT--4o--mini%20%7C%20OpenRouter-orange.svg)](https://openrouter.ai/)

An advanced **Retrieval-Augmented Generation (RAG)** web application
designed specifically for **academic research materials** and
**university lecture notes**.

## 📌 Key Features

-   **Hybrid Retrieval:** BM25 + FAISS + Reciprocal Rank Fusion (RRF)
-   **Grounded Generation:** GPT-4o-mini answers strictly from uploaded
    documents.
-   **Inline Citations:** File names and page numbers included
    automatically.
-   **Secure API Management:** Uses `st.secrets` for deployment.
-   **Performance Analytics:** Retrieval and generation latency metrics.

------------------------------------------------------------------------

## 🏗️ System Architecture

``` text
Uploaded PDFs
      │
      ▼
PDF Extraction + Metadata
      │
      ▼
Text Chunking (300 words, 50 overlap)
      │
 ┌────┴────┐
 ▼         ▼
BM25     FAISS
 └────┬────┘
      ▼
     RRF
      ▼
Top-K Chunks
      ▼
GPT-4o-mini
      ▼
Answer + Page Citations
```

## 🛠️ Tech Stack

  Component         Technology
  ----------------- ----------------------------------------
  Frontend          Streamlit
  Backend           Python 3.10+
  PDF Processing    pdfplumber / pypdf
  Embeddings        sentence-transformers/all-MiniLM-L6-v2
  Vector Database   FAISS
  Lexical Search    rank-bm25
  LLM               GPT-4o-mini via OpenRouter
  Deployment        Streamlit Cloud

## 🚀 Getting Started

``` bash
git clone https://github.com/your-username/academic-hybrid-rag.git
cd academic-hybrid-rag

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

## ☁️ Streamlit Cloud Secrets

``` toml
OPENROUTER_API_KEY="sk-or-v1-your-api-key"
OPENROUTER_MODEL="openai/gpt-4o-mini"
```

## 📐 Reciprocal Rank Fusion

\[ RRF(d)=`\sum`{=tex}\_{m`\in `{=tex}M}`\frac{1}{k+r_m(d)}`{=tex} \]

Where: - **M**: Set of retrievers (BM25 + FAISS) - **rₘ(d)**: Rank of
document *d* - **k**: Smoothing constant (typically 60)

## ✅ Features Checklist

-   [x] Hybrid Retrieval (BM25 + FAISS + RRF)
-   [x] Inline page citations
-   [x] Secure API handling
-   [x] Streamlit Cloud deployment
-   [x] Modular Python project

## 👨‍💻 Author

**Ahmed Abdelaziz Atallah**

Data Analyst | AI Graduate Python • SQL • Power BI • Excel • Machine Learning Microsoft Certified: Power BI Data Analyst (PL-300)
