# StrOutputParser is default for string manipulation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os 
from dotenv import load_dotenv

# Loading env variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Monitoring on Langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question: {question}")
    ]
)

# Streamlit init
st.title("Chatbot powered by Langchain and OpenAI")
input_text = st.text_input("Ask me anything")

# OpenAI LLM model
llm = ChatOpenAI(model = "gpt-4o-mini")

# Output parser
output_parser = StrOutputParser()

# Building the chain
chain = prompt|llm|output_parser

# Displaying the output on streamlit app
if input_text:
    st.write(chain.invoke({"question": input_text}))

