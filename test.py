import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from pinecone import Pinecone, ServerlessSpec
import time
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize Flask app
app = Flask(__name__)

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
    
    # Retrieve the top 3 most similar documents along with their scores
    similar_docs = vectorstore.similarity_search_with_score(query, k=5)
    
    # Filter out contexts with a similarity score below 0.3 and remove duplicates
    unique_docs = []
    seen_contents = set()
    for doc, score in similar_docs:
        if doc.page_content not in seen_contents:
            unique_docs.append((doc, score))
            seen_contents.add(doc.page_content)
    
    contexts = []
    for doc, score in unique_docs:
        contexts.append({
            'context': doc.page_content,
            'score': score
        })
    
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
        'contexts': contexts  # Return the list of contexts with scores
    })

@app.route('/diagram')
def diagram():
    return render_template('display_image.html')

if __name__ == '__main__':
    app.run(debug=True)

