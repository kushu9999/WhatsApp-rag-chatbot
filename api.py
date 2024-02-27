from fastapi import FastAPI, Form, Request
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community import embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from twillio import send_reply
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_ENDPOINT = os.getenv('GROQ_API_ENDPOINT')
GROQ_MODEL = os.getenv('GROQ_MODEL')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "API Running 200 OK"}

@app.post("/twilio")
async def twilio(request: Request, Body: str = Form(...), From: str = Form(...)):
    try:
        query = Body
        sender_id = From
        print(sender_id, query)

        vectorstore_path = "./models/lavender-models/faiss_index"

        # embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text')
        embedding=OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

        # load vectorstore from local
        vector_store = FAISS.load_local(vectorstore_path, embedding)

        # getting llm
        llm = ChatOpenAI(base_url=GROQ_API_ENDPOINT, model=GROQ_MODEL, api_key=GROQ_API_KEY)

        # Retriver Chain
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 4}))

        # retrive results from llm
        result = qa.run(query)

        print(result)

        # reply back to user
        send_reply(sender_id,result)

    except Exception as e:
        result = f"Something went wrong. Error: {e}"
        send_reply(sender_id,result)

    return {"message": "Message sent sucessfully"}
