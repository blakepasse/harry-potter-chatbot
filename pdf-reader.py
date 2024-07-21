from pypdf import PdfReader

reader = PdfReader("sample.pdf")
page = reader.pages[0]
print(page.extract_text())