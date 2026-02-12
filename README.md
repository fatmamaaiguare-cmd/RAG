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


  2. Installation
Clone this repository and install the required Python packages:

Bash
git clone [https://github.com/fatmamaaiguare-cmd/RAG.git](https://github.com/fatmamaaiguare-cmd/RAG.git)
cd RAG
pip install streamlit langchain langchain-community chromadb sentence-transformers pypdf
3. Run the App
Bash
streamlit run app.py
🧠 How It Works
Document Loading: The user uploads a PDF which is processed by PyPDFLoader.

Recursive Splitting: The text is broken into 1000-character chunks with 200-character overlap to maintain semantic continuity.

Vectorization: Chunks are converted into high-dimensional vectors using HuggingFace's all-MiniLM-L6-v2.

Contextual Retrieval: When a question is asked, the system retrieves the top 6 most relevant chunks from ChromaDB.

Strict Inference: The prompt forces the LLM to use only the retrieved context. If the answer isn't in the document, it will explicitly say so.

📁 Project Structure
app.py: Main Streamlit application and RAG logic.

requirements.txt: List of dependencies.

README.md: Project documentation.

Created as part of my Master's Degree in AI & Data Analytics.
