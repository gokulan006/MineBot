from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from src.prompt import SYSTEM_PROMPT, CONTEXTUALIZE_Q_N_SYSTEM_PROMPT
from src.helper import get_session_history

##
# Load env vars
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'


# Flask app
app = Flask(__name__)
app.secret_key=os.getenv("SECRET_KEY")


users = {}
# Global store (for session-wise chat memory)
chat_store = {}

# Initialize components
model = ChatGroq(model='gemma2-9b-it', api_key=groq_api)
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectordb = Chroma(persist_directory='./chromadb', embedding_function=embedding)
retriever = vectordb.as_retriever()

# Prompts
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ('system', CONTEXTUALIZE_Q_N_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ('human', '{input}')
])

qa_prompt = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ('human', '{input}')
])

output_parser=StrOutputParser()
# Chains
history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
qa_chain = create_stuff_documents_chain(model, qa_prompt)
qa_chain=qa_chain|output_parser
retrieval_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

 
rag_chain = RunnableWithMessageHistory(
    retrieval_chain,
    lambda session_id: get_session_history(session_id, chat_store),
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


import pandas as pd
from datetime import datetime

# Add this near the top of your app.py
NEWS_CSV_PATH = 'news_articles.csv'   

def load_news_articles():
    try:
        # Read CSV file
        df = pd.read_csv(NEWS_CSV_PATH)
        
        # Convert to list of dictionaries
        articles = df.to_dict('records')
        
        # Convert date strings to datetime objects for sorting
        for article in articles:
            article['published_date'] = datetime.strptime(article['published_date'], '%Y-%m-%d').date()
        
        # Sort by date (newest first)
        articles.sort(key=lambda x: x['published_date'], reverse=True)
        
        return articles
    except Exception as e:
        print(f"Error loading news articles: {e}")
        return []

 
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('chat'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            flash('Username already exists!', 'danger')
            return redirect(url_for('signup'))

        users[username] = generate_password_hash(password)
        flash('Signup successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user_hash = users.get(username)
        if user_hash and check_password_hash(user_hash, password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('chat'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

# @app.route('/chat')
# def chat():
#     if 'username' not in session:
#         return redirect(url_for('login'))
#     return render_template('index.html', username=session['username'])
  
@app.route('/chat')
def chat():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_input = data.get("message")
    session_id = data.get("session_id")
    language = data.get("language", "en")  # Default to English if not specified
    
    # Modify the input based on language
    if language == "hi":
        user_input = f"Answer this question in Hindi: {user_input}"
    

    try:
        result = rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
         
        # output_parser(result)
        return jsonify({
            "answer": result['answer'] 
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route("/news")
def news_page():
    articles = load_news_articles()
    return render_template("news.html", articles=articles)

if __name__ == "__main__":

    app.run(debug=True,threaded=False)
