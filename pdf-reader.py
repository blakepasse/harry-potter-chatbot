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
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob
from dotenv import load_dotenv

loader = TextLoader('/Users/bpasse/Desktop/virtual-tests/project/converted/extracted_text.txt')

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

for doc in docs:
    print(f"{doc}")
    print("\n\n")

load_dotenv()

oapi = os.getenv("OPENAI_API_KEY")
papi = os.getenv("PINECONE_API_KEY")

os.environ['OPENAI_API_KEY'] = oapi
os.environ['PINECONE_API_KEY'] = papi

