import PyPDF2

def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

pdf_path = 'C:/Users/Tianjian/OneDrive - UNSW/桌面/EdTech Roadmap 2024-2028 Full report.pdf'
pdf_text = read_pdf(pdf_path)
