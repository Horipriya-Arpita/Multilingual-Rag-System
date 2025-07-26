# 🤖 Multilingual RAG System (Bangla + English)

A simple **Multilingual Retrieval-Augmented Generation (RAG)** system built to answer both **Bengali and English** queries from a scanned Bengali textbook (HSC Bangla 1st Paper). It uses **OCR**, **vector search**, and **LLM-based response generation** to answer questions from documents.

---

## 📌 Features

- 🔍 Accepts Bangla and English questions.
- 📚 Extracts content from scanned Bengali PDF using OCR.
- 🧹 Cleans and chunks the text with paragraph-aware logic.
- 🧠 Embeds and indexes with multilingual sentence transformers.
- 🤖 Answers questions with context using **Gemini** (Google Generative AI).
- 🔗 Includes REST API using FastAPI.
- 🧪 Includes manual evaluation + question relevance tracking.

---

## 📁 Folder Structure

```
Multilingual-RAG/
├── src/
│   ├── main.py               # OCR Extraction (Tesseract)
│   ├── preprocess.py         # Cleaning
│   ├── embed_chunks.py       # Chunking + FAISS
│   ├── rag.py                # Core Retrieval + LLM logic
│   ├── api.py                # FastAPI endpoint
├── data/
│   ├── hsc.pdf               # (Bengali book - not uploaded)
│   ├── processed_text.txt
│   ├── cleaned_text.txt
│   ├── sample_queries.md
│   └── faiss/
│       ├── index.bin
│       └── chunks.pkl
├── .env.example              # Place your GEMINI_API_KEY here
├── requirements.txt
├── README.md
├── project_answers.md
├── evaluation.md
```

---

## 🚀 Setup Instructions

### 1️⃣ Clone & Setup Environment

```bash
git clone https://github.com/your-username/Multilingual-RAG.git
cd Multilingual-RAG

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2️⃣ Add Gemini API Key

Create `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

### 3️⃣ Run Preprocessing Steps

```bash
python src/main.py          # OCR
python src/preprocess.py    # Clean text
python src/embed_chunks.py  # Chunk & embed text
```

---

## 🧪 Run the System

### CLI Interface
```bash
python src/rag.py
```

### API Interface
```bash
uvicorn src.api:app --reload
```

---

## 🔗 API Documentation

**POST** `/ask`

**Request Body:**
```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

**Response:**
```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "শুম্ভুনাথ"
}
```

---

## 📝 Sample Queries

| User Question | Expected Answer |
|---------------|-----------------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | শুম্ভুনাথ |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর |

---

## ✅ Evaluation Summary

| Question | Retrieved Context Correct? | Answer Match | Comments |
|----------|----------------------------|--------------|----------|
| সুপুরুষ | ✅ | ✅ | Exact match |
| ভাগ্য দেবতা | ✅ | ✅ | Match |
| কল্যাণীর বয়স | ✅ | ✅ | Accurate |

---

## ❓ Answers to Assignment Questions

### Q1: What method/library did you use to extract the text, and why?

> I used `pytesseract` with `pdf2image` to handle the scanned PDF. Since it's an image-based document (not digital text), OCR is necessary. Tesseract with the Bengali language pack (`lang='ben'`) provided decent accuracy. Preprocessing was required to fix common OCR issues.

---

### Q2: What chunking strategy did you choose?

> I used `RecursiveCharacterTextSplitter` with `chunk_size=300` and `chunk_overlap=50`, splitting on Bengali-specific punctuation like `।`, as well as `\n`. This gives semantic chunks while keeping the FAISS vector space effective.

---

### Q3: What embedding model did you use and why?

> I used `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which supports over 50 languages including Bangla. It provides strong semantic embeddings and is lightweight enough for local experimentation.

---

### Q4: How do you compare query with stored chunks?

> I use **FAISS** with L2 distance to find top-k most similar vectors. The same embedding model is used for both queries and document chunks to ensure uniform semantic space.

---

### Q5: How do you ensure meaningful comparison?

> Since both the query and chunks are embedded using the same multilingual transformer, they map to a shared latent space. If a query is vague, the system may retrieve general or unrelated results — better chunking or larger corpus could help mitigate that.

---

### Q6: Do the results seem relevant? If not, how to improve?

> Yes, the results are generally relevant and grounded. However, I believe they could be even better if I had access to OpenAI models — I couldn’t use them due to a lack of credits/resources. If I could work with GPT-4 or similar, the output would likely be more accurate and insightful.  
Additionally, I wasn’t able to use LangChain fully due to time and resource limitations. If I had used it, the system could have benefited from better document management, chunking strategies, and prompt optimization.  
Future improvements could include:
> - Sentence-level chunking for cleaner context
> - Larger context windows in prompts  
> - Using advanced models like Gemini Pro or GPT-4 
> - Associating metadata with chunks for better filtering 

---

## 🛠️ Tools & Libraries Used

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [pdf2image](https://pypi.org/project/pdf2image/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence-Transformers](https://www.sbert.net/)
- [Gemini API (Google Generative AI)](https://ai.google.dev/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Python-dotenv](https://pypi.org/project/python-dotenv/)
- [langdetect](https://pypi.org/project/langdetect/)

---

## ✅ Final Notes

- Works on scanned Bengali PDFs and supports multilingual queries
- Clean, modular code with API integration
- Easy to extend with frontend or chatbot interface

---

> Built for the **AI Engineer (Level-1) Technical Assessment**  
> Author: Horipriya Das Arpita   
> Contact: horipriya288@gmail.com  
> Github: https://github.com/Horipriya-Arpita  
