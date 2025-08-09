import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
 

# Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","Hey you are an intelligent AI assistant. Please answer to my question upto your best."),
        ("user","Question:{input}")
    ]
)

# Get Response Function
def get_response(input,model,temperature,max_tokens):
    llm=ChatGroq(model=model)
    chain=prompt|llm|StrOutputParser()
    answer=chain.invoke({'input':input})
    return answer

st.title("Q&A Chatbot with Open Source LLM Models")

st.sidebar.title("Settings")
model=st.sidebar.selectbox("Select an Open Source LLM model",['gemma2-9b-it','llama-3.3-70b-versatile','deepseek-r1-distill-llama-70b' ,'compound-beta'])
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=1500,value=150)

st.write("Go ahead and ask any query")
input=st.text_input("You:")


if input:
    response=get_response(input,model,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide a query")

