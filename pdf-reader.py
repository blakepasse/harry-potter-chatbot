import os
from pypdf import PdfReader

# Extract text from pdfs
pdf_path = "/Users/bpasse/Desktop/virtual-tests/project/documents/sample-newspaper.pdf"

output_dir = "/Users/bpasse/Desktop/virtual-tests/project/converted"
output_file = "extracted_text.txt"

output_path = os.path.join(output_dir, output_file)

reader = PdfReader(pdf_path)
extracted_text = ""
for page in reader.pages:
    extracted_text += page.extract_text()

with open(output_path, "w") as text_file:
    text_file.write(extracted_text)


with open("/Users/bpasse/Desktop/virtual-tests/project/converted/extracted_text.txt") as f:
    text = f.read()

# Chunk text
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([text])

# Embed chunks in pinecone
# Use chatbot