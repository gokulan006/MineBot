from langchain_core.chat_history import BaseChatMessageHistory 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

# HuggingFace token
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
def get_session_history(session_id:str,store)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

def create_embedding_fnc(model_name='all-MiniLM-L6-v2'):
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

def load_split_documents(pdf_path):
    doc=PyPDFLoader(pdf_path).load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    document=splitter.split_documents(doc)
    return document