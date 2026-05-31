
import os
from dotenv import load_dotenv
import streamlit as st

# LangChain Imports
from langchain_openai import ChatOpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain

from langchain_classic.memory import ConversationBufferMemory

# Load Environment Variables
load_dotenv()

# API Key
XAI_API_KEY = os.getenv("XAI_API_KEY")

# Streamlit Config
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    text-align: center;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 AI PDF RAG Chatbot")

st.markdown(
    "<h4 style='text-align:center;'>Upload a PDF and chat with your document</h4>",
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:

    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF File",
        type="pdf"
    )

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Process PDF
if uploaded_file and st.session_state.qa_chain is None:

    with st.spinner("Processing PDF..."):

        # Save PDF
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader("temp.pdf")

        documents = loader.load()

        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        texts = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector Store
        vectorstore = FAISS.from_documents(
            texts,
            embeddings
        )

        # Retriever
        retriever = vectorstore.as_retriever()

        # LLM
        llm = ChatOpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
            model="grok-4.20-reasoning",
            temperature=0
        )

        # Memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # QA Chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )

        # Store Chain
        st.session_state.qa_chain = qa_chain

        st.success("✅ PDF Uploaded Successfully")

# Display Previous Messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
user_question = st.chat_input(
    "Ask questions from your PDF..."
)

# User Question
if user_question:

    # Check PDF Upload
    if st.session_state.qa_chain is None:

        st.error("Please upload a PDF first.")

    else:

        # Save User Message
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        # Show User Message
        with st.chat_message("user"):
            st.markdown(user_question)

        # AI Response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    response = st.session_state.qa_chain.invoke({
                        "question": user_question
                    })

                    answer = response["answer"]

                except Exception as e:

                    answer = f"Error: {str(e)}"

                st.markdown(answer)

        # Save Assistant Message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })