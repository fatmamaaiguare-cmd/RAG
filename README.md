# 🤖 Local RAG Chatbot (Ollama + LangChain)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-121212.svg)](https://www.langchain.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-white.svg)](https://ollama.com/)

A professional **Retrieval-Augmented Generation (RAG)** application built to chat with PDF documents locally. This project prioritizes **data privacy** and cost-efficiency by running everything—from embeddings to the LLM—on your own hardware.

## 🌟 Key Features
- **100% Private & Local:** No external API calls (No OpenAI/Anthropic keys needed). Your data never leaves your machine.
- **Advanced Document Processing:** Uses `RecursiveCharacterTextSplitter` for optimal context preservation.
- **Efficient Vector Search:** Powered by **ChromaDB** and HuggingFace's `all-MiniLM-L6-v2` embeddings for fast, CPU-friendly retrieval.
- **Strict Response Mode:** The AI is programmed to answer *only* based on the provided document, preventing hallucinations.
- **Real-time Streaming:** Features a smooth, modern chat interface with streaming text responses.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **LLM Orchestration:** LangChain
- **Local LLM:** Ollama (defaulting to **Phi-3** or **Mistral**)
- **Embeddings:** HuggingFace Transformers
- **Vector Store:** ChromaDB

## 🚀 Getting Started

### 1. Prerequisites
- Install [Ollama](https://ollama.com/).
- Pull the desired model (this app uses `phi3` by default):
  ```bash
  ollama pull phi3
