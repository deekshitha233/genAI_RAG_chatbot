# 🤖 Conversational Multi-Document RAG Chatbot

An AI-powered Conversational Retrieval-Augmented Generation (RAG) chatbot that allows users to upload multiple PDF documents and ask questions about their content.

Built using:
- Streamlit
- OpenAI / xAI Compatible APIs
- LangChain
- FAISS Vector Database
- RAG Architecture
- PDF Processing

---

## 🚀 Features

✅ Upload multiple PDF documents

✅ Conversational chat interface

✅ Retrieval-Augmented Generation (RAG)

✅ Semantic search using embeddings

✅ Vector storage using FAISS

✅ Context-aware responses

✅ Chat history support

✅ Modern Streamlit UI

✅ Deployable on Hugging Face Spaces

---

## 🏗️ Project Architecture

```text
User Uploads PDFs
        │
        ▼
PDF Text Extraction
        │
        ▼
Text Chunking
        │
        ▼
Embedding Generation
        │
        ▼
FAISS Vector Database
        │
        ▼
Retriever
        │
        ▼
LLM (OpenAI/xAI)
        │
        ▼
AI Response
