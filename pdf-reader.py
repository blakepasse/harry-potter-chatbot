import os
from pypdf import PdfReader

pdf_path = "/Users/bpasse/Desktop/virtual-tests/project/documents/sample-newspaper.pdf"

# Define the directory and file name for the output text file
output_dir = "/Users/bpasse/Desktop/virtual-tests/project/converted"
output_file = "extracted_text.txt"

# Define the full path to the output text file
output_path = os.path.join(output_dir, output_file)

# Read the PDF and extract text
reader = PdfReader(pdf_path)
extracted_text = ""
for page in reader.pages:
    extracted_text += page.extract_text()

# Write the extracted text to the output file
with open(output_path, "w") as text_file:
    text_file.write(extracted_text)


# This is a long document we can split up.
with open("/Users/bpasse/Desktop/virtual-tests/project/converted/extracted_text.txt") as f:
    text = f.read()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([text])