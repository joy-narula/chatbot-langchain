from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import streamlit as st
import os
import time


from dotenv import load_dotenv

load_dotenv()

# load api keys for groq and openai
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.environ['GROQ_API_KEY']

st.title("ChatGroq with Llama3 for Resume Understanding")

llm = ChatGroq(groq_api_key = groq_api_key, 
               model="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context.
Please provide the most approximate response based on the question, if something is not explicitly mentioned please use approximation and smart judgement using similarlities.
Pretend and be confident in your answer even though it may seem like an approximate ans.
Please provide answer as facts and statements in no more than 2 lines

<context>
{context}
<context>

Question: {input}

"""
)

def vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()

        # Data Ingestion
        st.session_state.loader = PyPDFDirectoryLoader("./resumes")
        
        # Document loading
        st.session_state.docs = st.session_state.loader.load()
        
        # Chunk Creation
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Vector store using OpenAI embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)


prompt1 = st.text_input("Enter your questions about the candidate resume")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB is ready")



if prompt1:
    # creates context from documents
    document_chain = create_stuff_documents_chain(llm, prompt)

    # interface for vector db
    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print(f"Response time: {time.process_time() - start}")
    st.write(response['answer'])

    # Using expander to display the context
    with st.expander("Document Similarlity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---------------------------------------------")
    