from pypdf import PdfReader

reader = PdfReader("/Users/bpasse/Desktop/virtual-tests/project/documents/sample-newspaper.pdf")
page = reader.pages[2]
print(page.extract_text())