import ebooklib
from ebooklib import epub
from io import BytesIO
from reportlab.pdfgen import canvas
from epub_conversion.utils import open_book, convert_epub_to_lines


# def epub_to_pdf(epub_file_path):
#     book = open_book(epub_file_path)
#     lines = convert_epub_to_lines(book)
#     return lines


from ebooklib import epub
from io import BytesIO
from reportlab.pdfgen import canvas
from epub_conversion.utils import open_book, convert_epub_to_lines

def epub_to_pdf(epub_file_path):
    book = open_book(epub_file_path)
    lines = convert_epub_to_lines(book)
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer)
    for line in lines:
        c.drawString(10, 800, line)
        c.showPage()
    c.save()
    return pdf_buffer

import PyPDF2
import io

pdf_buffer = epub_to_pdf("docs/Breakthrough_Advertising_-_Eugene_Schwartz.epub")

# Create a pdf reader object from the pdf buffer
pdf_reader = PyPDF2.PdfReader(pdf_buffer)

# Create a pdf writer object for the output file
pdf_writer = PyPDF2.PdfWriter()

# Loop through all the pages and add them to the writer object
for page_number in range(len(pdf_reader.pages)):
    pdf_writer.add_page(pdf_reader.pages[page_number])

# Create a text file object for the output file
text_file = open("docs/output.txt", "w", encoding="utf-8")


import re

# Define a pattern to match all html or xml tags
pattern = re.compile('<[^<]+?>')

# Define a function to remove the tags from a string
def remove_tags(text):
    # Use the sub() method to replace the tags with an empty string
    cleantext = re.sub(pattern, '', text)
    # Return the cleantext
    return cleantext


# Loop through all the pages and extract the text
for page_number in range(len(pdf_writer.pages)):
    page_obj = pdf_writer.pages[page_number]
    text = page_obj.extract_text()
    # Write the text to the text file
    text_file.write(remove_tags(text).replace('■', ''))

# Close the text file
text_file.close()

