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

print(f"Text extracted and saved to {output_path}")