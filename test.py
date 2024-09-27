import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from pinecone import Pinecone, ServerlessSpec
import time
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from werkzeug.security import generate_password_hash, check_password_hash
from collections import OrderedDict

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this to a random secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load environment variables
load_dotenv()
oapi = os.getenv("OPENAI_API_KEY")
papi = os.getenv("PINECONE_API_KEY")
os.environ['OPENAI_API_KEY'] = oapi
os.environ['PINECONE_API_KEY'] = papi

# Initialize Pinecone using the Pinecone class
pc = Pinecone(api_key=papi)

# Define the index name and namespace
index_name = "rag-project"
namespace = "harry_potter"

# Ensure the index exists
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-west-2')
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Create the embedding model
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# Create the vector store instance with the existing index and namespace
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding,
    namespace=namespace,
)

# Define the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

from langchain.chains import RetrievalQA

# Set up the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template="""You are a helpful Harry Potter assistant. Your task is to answer questions based on the provided context.
    If the question is about who you are, respond with "I am your helpful Harry Potter assistant."
    For all other questions, use the following context to formulate your answer:
    
    Context: {context}
    
    Question: {query}
    
    Answer:"""
)

# Create an LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['query']
    print(f"Received query: {query}")  # Debugging line
    
    # Retrieve the top 5 most similar documents along with their scores
    similar_docs = vectorstore.similarity_search_with_score(query, k=5)
    
    # Use OrderedDict to maintain order and ensure uniqueness
    unique_contexts = OrderedDict()
    for doc, score in similar_docs:
        # Use the context as the key to ensure uniqueness
        if doc.page_content not in unique_contexts:
            unique_contexts[doc.page_content] = {
                'context': doc.page_content,
                'score': score
            }
    
    # Convert the OrderedDict values back to a list
    contexts = list(unique_contexts.values())
    
    # Use the most relevant context as part of the input to the LLM
    if contexts:
        context = contexts[0]['context']
        response_with_knowledge = llm_chain.run(query=query, context=context)
        print(f"Response: {response_with_knowledge}")  # Debugging line
        response_text = response_with_knowledge
    else:
        # Handle case where no relevant context is found
        response_text = llm_chain.run(query=query, context="No relevant context was found in the database.")
        print(f"No relevant context found. Generated response: {response_text}")
    
    return jsonify({
        'response': response_text,
        'contexts': contexts,
        'id': str(time.time())  # Add a unique ID for each response
    })

@app.route('/rank', methods=['POST'])
@login_required
def rank():
    data = request.json
    response_id = data['id']
    ranking = data['ranking']
    # Here you would typically store the ranking in a database
    # For now, we'll just print it
    print(f"Response {response_id} received ranking: {ranking}")
    return jsonify({'status': 'success'})

@app.route('/diagram')
def diagram():
    return render_template('display_image.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        return 'Invalid username or password'
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
