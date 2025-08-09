import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import SystemMessage
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

 
os.environ.pop("SSL_CERT_FILE", None)

load_dotenv()
groq_api=os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')

st.title('Mining Regulations Assitant')

st.sidebar.title('Settings')
groq_api=st.sidebar.text_input("Enter Groq API Key",value=groq_api,type='password')


pdf_upload=st.sidebar.checkbox("Want to explore any pdf")
model=ChatGroq(model='gemma2-9b-it',api_key=groq_api)
embedding=HuggingFaceEmbeddings(
        model='sentence-transformers/all-MiniLM-L6-v2')

if pdf_upload:
    
    session_id=st.text_input("Enter Session ID",value='chat1')
    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files=st.file_uploader("Upload a pdf File",type='pdf',accept_multiple_files=True)
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
        splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
        final_document=splitter.split_documents(documents)
        vectordb=Chroma.from_documents(final_document,embedding)
        retriever=vectordb.as_retriever()

        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a stand alone question which can be understood"
            "without the chat history. Do NOT answer the question,"
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',contextualize_q_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}')
            ]
        )
        system_prompt="""
            You are an intelligent AI assistant.
            Use the following pieces of retrived context to answer.
            If you don't the answer say that you don't know.
            Use three sentences maximum and keep your answer concise.
            \n\n
            {context}.
        """
        q_a_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}')
            ]
        )
        history_aware_retriever=create_history_aware_retriever(model,retriever,contextualize_q_prompt)
        question_answer_chain=create_stuff_documents_chain(model,q_a_prompt)
        retrieval_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        rag_chain=RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )
        input=st.text_input("Ask Question from the documents uploaded")
        if input:
            response=rag_chain.invoke(
                {"input":input},
                config={
                    "configurable":{"session_id":session_id}
                }
            )
            st.success(response['answer'])
            session_history=get_session_history(session_id)
            with st.expander('Context Document'):
                st.write(response['context'])
            with st.expander("Chat History"):
                st.write(session_history.messages)


else:  
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sample Mining Questions:**")
    st.sidebar.markdown("- What are the safety regulations for underground mining?")
    st.sidebar.markdown("- How does mining impact local water sources?")
    st.sidebar.markdown("- What are the latest technologies for sustainable mining?")

    SYSTEM_PROMPT="""
    You are a Mining Regulation Assistant focused exclusively on:

    Mining regulations/compliance (OSHA, MSHA, EPA)
    Safety standards
    Environmental impacts
    Extraction methods
    Mine rehabilitation

    Rules:
    Only answer mining-related questions.

    For off-topic queries, respond:
    'I specialize in mining topics only. Ask about regulations, safety, or environmental impacts in mining.'

    Provide:
    Clear, factual answers
    Relevant regulations (cite sources if possible)
    Practical implications
    """
    prompt=ChatPromptTemplate.from_messages(
        [
            ('system',SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name='messages')
        ],
        
    )

    output_parser=StrOutputParser()

    trimmer=trim_messages(
        max_tokens=6000,
        token_counter=model,
        strategy='last',
        start_on='human',
        include_system=True,
        allow_partial=False,
    )

    chain=(
        RunnablePassthrough.assign(messages=itemgetter("messages")|trimmer)
        | prompt
        | model
        | output_parser
    )


    store={}
    def get_session_history(session_id:str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id]=ChatMessageHistory()
        return store[session_id]


    with_message_history=RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key='messages'
    )

    config={'configurable':{'session_id':'chat1'}}
    st.write("Ask questions about mining regulations, safety standards, environmental impacts, and industry best practices. I'll provide accurate, up-to-date information.")
    message=st.text_input('Ask a query')

    if message:
        response=with_message_history.invoke(
            {"messages":message},
            config=config
        )
        st.success(response)

