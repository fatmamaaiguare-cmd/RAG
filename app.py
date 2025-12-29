import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Titre et Configuration ---
st.set_page_config(page_title="Assistant RAG Local", layout="wide")
st.title("🤖 Chatbot 100% Local (Ollama + Mistral)")
st.markdown("Interrogation de document sans API Key payante.")

# --- 2. Chargement et Vectorisation ---
@st.cache_resource
def setup_vector_db(file_path):
    # A. Chargement
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # B. Découpage
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    # C. Embeddings Gratuits (HuggingFace tourne sur votre CPU)
    # Le modèle 'all-MiniLM-L6-v2' est très rapide et léger
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # D. Base Vectorielle
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# Gestion de l'upload fichier
uploaded_file = st.file_uploader("Chargez un PDF", type="pdf")

if uploaded_file:
    # Création d'un fichier temporaire pour gérer le PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    with st.spinner("Analyse du document en local (cela peut prendre un moment)..."):
        vectorstore = setup_vector_db(tmp_path)
        # k=7 signifie : "Donne-moi les 7 passages les plus pertinents du PDF"
# Au lieu de 4 par défaut. Cela aide pour la synthèse.
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6}) 
    st.success("Document prêt ! Posez vos questions.")

    # --- 3. Configuration du LLM Local (Ollama) ---
    # Assurez-vous d'avoir lancé 'ollama run mistral' dans votre terminal avant !
    llm = ChatOllama(model="phi3", temperature=0)

    # Prompt
   # Prompt Système (Version Stricte)
    system_prompt = (
        "Tu es un assistant spécialisé qui répond UNIQUEMENT en se basant sur le contexte fourni ci-dessous. "
        "Il est INTERDIT d'utiliser tes connaissances générales ou externes. "
        "Si la réponse n'est pas explicitement écrite dans le contexte, réponds exactement : 'Je ne trouve pas cette information dans le document.' "
        "Ne tente pas d'inventer. Sois direct."
        "\n\n"
        "Contexte : {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # --- 4. Interface Chat ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Votre question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response_placeholder = st.empty() # Créer un conteneur vide
            full_response = ""
            
            # On utilise .stream() au lieu de .invoke()
            try:
                # On boucle sur les morceaux de réponse qui arrivent
                for chunk in rag_chain.stream({"input": user_input}):
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        # Mise à jour dynamique de l'affichage avec un petit curseur
                        response_placeholder.markdown(full_response + "▌")
                
                # Affichage final propre sans le curseur
                response_placeholder.markdown(full_response)
                
                # Sauvegarde dans l'historique
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Erreur : {e}")

else:
    st.info("Veuillez charger un PDF pour activer le chat.")