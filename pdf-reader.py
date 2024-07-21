# import os
# from pypdf import PdfReader

# # Extract text from pdfs
# pdf_path = "/Users/bpasse/Desktop/virtual-tests/project/documents/sample-newspaper.pdf"

# output_dir = "/Users/bpasse/Desktop/virtual-tests/project/converted"
# output_file = "extracted_text.txt"

# output_path = os.path.join(output_dir, output_file)

# reader = PdfReader(pdf_path)
# extracted_text = ""
# for page in reader.pages:
#     extracted_text += page.extract_text()

# with open(output_path, "w") as text_file:
#     text_file.write(extracted_text)


# with open("/Users/bpasse/Desktop/virtual-tests/project/converted/extracted_text.txt") as f:
#     text = f.read()

# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()


# oapi = os.getenv("OPENAI_API_KEY")

# chat = ChatOpenAI(
#     openai_api_key=oapi,
#     model='gpt-3.5-turbo'
# )

import magic
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob
from dotenv import load_dotenv
load_dotenv()
oapi = os.getenv("OPENAI_API_KEY")
papi = os.getenv("PINECONE_API_KEY")
os.environ['OPENAI_API_KEY'] = oapi
os.environ['PINECONE_API_KEY'] = papi

loader = TextLoader('/Users/bpasse/Desktop/virtual-tests/project/converted/extracted_text.txt')

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

split_docs_strings = [doc.page_content for doc in split_docs]

embedding = OpenAIEmbeddings(
    model = "text-embedding-3-small",
)

index_name = "vid-chatbot"
namespace = "new"
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