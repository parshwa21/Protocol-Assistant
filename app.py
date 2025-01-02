import os
import PyPDF2
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json

# User authentication data (can be replaced with a database or secure auth mechanism)
USER_DATA_FILE = "users.json"

# Functions for authentication
def load_user_data():
    if not os.path.exists(USER_DATA_FILE):
        return {}
    with open(USER_DATA_FILE, "r") as file:
        return json.load(file)

def authenticate(username, password):
    users = load_user_data()
    return username in users and users[username] == password

# PDF processing functions
def list_pdfs(directory):
    """List all PDF files in the specified directory."""
    return [f for f in os.listdir(directory) if f.endswith(".pdf")]

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF."""
    with open(pdf_file, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def create_qa_chain(pdf_text):
    """Create a QA chain using LangChain."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(pdf_text)
    docs = [Document(page_content=text) for text in texts]

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    return ConversationalRetrievalChain.from_llm(
        llm=OpenAI(model="gpt-4"),
        retriever=vectorstore.as_retriever(),
    )

# Streamlit App
def main():
    st.title("Internal Study Protocol Assistant")
    
    # Login Section
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    logged_in = st.sidebar.button("Login")

    if logged_in and authenticate(username, password):
        st.sidebar.success(f"Welcome, {username}!")
        
        # File Selection
        pdf_directory = st.text_input("Enter the directory where PDFs are stored:")
        if os.path.exists(pdf_directory):
            pdf_files = list_pdfs(pdf_directory)
            if pdf_files:
                selected_pdf = st.selectbox("Select a PDF file:", pdf_files)
                if st.button("Process Selected PDF"):
                    pdf_path = os.path.join(pdf_directory, selected_pdf)
                    st.write(f"Processing {selected_pdf}...")
                    pdf_content = extract_text_from_pdf(pdf_path)
                    
                    qa_chain = create_qa_chain(pdf_content)
                    st.success("PDF processed! Ask your questions below.")
                    
                    # Chat Interface
                    chat_history = []
                    user_query = st.text_input("Ask a question about the document:")
                    if user_query:
                        response = qa_chain({"question": user_query, "chat_history": chat_history})
                        chat_history.append((user_query, response["answer"]))
                        st.write(f"**Answer:** {response['answer']}")
            else:
                st.warning("No PDFs found in the specified directory.")
        else:
            st.warning("Please enter a valid directory.")
    elif logged_in:
        st.sidebar.error("Invalid username or password.")

if __name__ == "__main__":
    main()
