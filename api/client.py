import requests
import streamlit as st

def get_openai_response(input_text):
    response = requests.post(
        "http://localhost:8000/openai_product/invoke",
        json = {'input': {'product': input_text}}
        
        )
    
    return response.json()['output']['content']

def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/openai_product/invoke",
        json = {'input': {'product': input_text}}
        
        )
    
    return response.json()['output']['content']

st.title('Langchain with OpenAI and Ollama')
input_openai = st.text_input("Which product from apple do you want info about, powered by openai")
input_llama = st.text_input("Which product from apple do you want info about, powered by llama3.2")
# Displaying the output on streamlit app
if input_openai:
    st.write(get_openai_response(input_openai))

# Displaying the output on streamlit app
if input_llama:
    st.write(get_ollama_response(input_llama))