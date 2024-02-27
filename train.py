# from utils import BedrockLLM
# from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
# from langchain.embeddings import BedrockEmbeddings
# from langchain_community import embeddings
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain_community.chat_models import ChatOllama
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# load PDF files from a directory
# loader = PyPDFDirectoryLoader("./data/rental-agreement")
loader = PyPDFLoader("./data/lavender.pdf")
data = loader.load()
# print the loaded data, which is a list of tuples (file name, text extracted from the PDF)
print(data)

# split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
text_chunks = text_splitter.split_documents(data)
# print the number of chunks obtained
print(len(text_chunks))


# AWS Bedrock embeddings
# bedrock_runtime_client = BedrockLLM.get_bedrock_client()
# embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock_runtime_client)

# ollama embeddings
# embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text')
embedding=OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)


# create embeddings for each text chunk using the FAISS class, which creates a vector index using FAISS and allows efficient searches between vectors
vector_store = FAISS.from_documents(text_chunks, embedding=embedding)

# save vectorstore
vectorstore_path = "./models/lavender-models/faiss_index"
vector_store.save_local(vectorstore_path)

# load vectorstore from local
vector_store = FAISS.load_local(vectorstore_path, embedding)

# getting llm
# llm = BedrockLLM.get_bedrock_llm()
# llm = ChatOllama(model="mistral-openorca")

# # Create a question answering system based on information retrieval using the RetrievalQA class, which takes as input a neural language model, a chain type and a retriever (an object that allows you to retrieve the most relevant chunks of text for a query)
# # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 4}))

# memory = ConversationBufferMemory(
#     memory_key='chat_history', return_messages=True)

# qa = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
#     memory=memory
# )
# # # define a query to ask the system
# # query = "What are payment terms in rental agreement?"
# # # run the system and get a response
# # print(qa.run(query))

# if __name__ == "__main__":
#     while True:
#         query = input("Please enter your question: ")
#         if query.lower() == "bye":
#             break

#         result = qa.run(query)
#         if result == 'IDK':
#             print("I don't have enough information to answer this question.")
#         else:
#             print("Answer:",result)