import pytesseract
from pdf2image import convert_from_path

# Set path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Use poppler
poppler_path = r"F:\Intern\poppler-24.08.0\Library\bin"

images = convert_from_path("documents/hsc.pdf", poppler_path=poppler_path)
text = ""
for img in images:
    text += pytesseract.image_to_string(img, lang='ben') + "\n"

# Save to file
with open("data/processed_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
