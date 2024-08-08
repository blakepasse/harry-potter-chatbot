import magic
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

load_dotenv()
oapi = os.getenv("OPENAI_API_KEY")
papi = os.getenv("PINECONE_API_KEY")
os.environ['OPENAI_API_KEY'] = oapi
os.environ['PINECONE_API_KEY'] = papi

loader = TextLoader('/Users/bpasse/Desktop/virtual-tests/project/converted/Harry_Potter_all_char_separated.txt')

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

split_docs_strings = [doc.page_content for doc in split_docs]

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

index_name = "rag-project"
namespace = "harry_potter"
vectorstore = PineconeVectorStore.from_texts(
    texts=split_docs_strings,
    index_name=index_name,
    embedding=embedding,
    namespace=namespace,
)

query = "People who are well rested do better on what?"

similar_docs = vectorstore.similarity_search(query)

print(similar_docs[0])

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

print(qa.invoke(query))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['query']
    print(f"Received query: {query}")  # Debugging line
    
    # Retrieve the top 3 most similar documents along with their scores
    similar_docs = vectorstore.similarity_search_with_score(query, k=3)
    
    contexts = []
    for doc, score in similar_docs:
        contexts.append({
            'context': doc.page_content,
            'score': score
        })
    
    # Use the most relevant context as part of the input to the LLM
    if contexts:
        response_with_knowledge = qa.invoke(query)
        print(f"Response: {response_with_knowledge}")  # Debugging line
        response_text = response_with_knowledge['result']
    else:
        response_text = "No relevant document found."
    
    return jsonify({
        'response': response_text,
        'contexts': contexts  # Return the list of contexts with scores
    })



@app.route('/diagram')
def diagram():
    return render_template('display_image.html')

if __name__ == '__main__':
    app.run(debug=True)
