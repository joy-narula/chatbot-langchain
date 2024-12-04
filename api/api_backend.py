from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

# Loading env variables
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

api = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="API Server"
)

add_routes(
    api,
    ChatOpenAI(),
    path="/openai"

)

llm_openai = ChatOpenAI()

llm_llama = Ollama(model="llama3.2")

prompt1 = ChatPromptTemplate.from_template("Explain me the working of {product} in 2 lines.")
prompt2 = ChatPromptTemplate.from_template("Explain me the working of {product} in 2 lines.")

add_routes(
    api,
    prompt1|llm_openai,
    path="/openai_product"

)

add_routes(
    api,
    prompt2|llm_openai,
    path="/llama_product"

)

if __name__ == "__main__":
    uvicorn.run(api, host="localhost", port=8000)

