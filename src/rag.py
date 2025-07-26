# import os
# import pickle
# import faiss
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI

# client = OpenAI()

# # Load API key
# load_dotenv()
# # openai.api_key = os.getenv("OPENAI_API_KEY")

# # Load FAISS and Chunks
# def load_faiss_index():
#     index = faiss.read_index("data/faiss/index.bin")
#     with open("data/faiss/chunks.pkl", "rb") as f:
#         chunks = pickle.load(f)
#     return index, chunks

# # Embed user query
# def embed_query(query, model):
#     return model.encode([query])

# # Retrieve top-k similar chunks
# def get_top_chunks(query_vec, index, chunks, k=3):
#     D, I = index.search(query_vec, k)
#     return [chunks[i] for i in I[0]]

# # Call OpenAI to generate answer
# def generate_answer(query, context):
#     prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"""

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",  # or gpt-4
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.2
#     )
#     return response.choices[0].message.content.strip()

# # Main
# def ask(query):
#     model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#     index, chunks = load_faiss_index()
#     query_vec = embed_query(query, model)
#     top_chunks = get_top_chunks(query_vec, index, chunks)
#     context = "\n".join(top_chunks)
#     answer = generate_answer(query, context)
#     return answer

# # CLI Testing
# if __name__ == "__main__":
#     while True:
#         user_q = input("\n🔍 Enter your query (Bangla/English): ")
#         if user_q.lower() in ["exit", "quit"]:
#             break
#         result = ask(user_q)
#         print(f"🤖 Answer: {result}")


import os
import pickle
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from langdetect import detect


# Load environment and Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load FAISS index and chunks
def load_faiss_index():
    index = faiss.read_index("data/faiss/index.bin")
    with open("data/faiss/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Embed user query
def embed_query(query, model):
    return model.encode([query])

# Retrieve top-k similar chunks
def get_top_chunks(query_vec, index, chunks, k=3):
    D, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

# Call Gemini API
# def generate_answer(query, context):
#     prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"""

    # model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
    # response = model.generate_content(prompt)
    # return response.text.strip()

def generate_answer(query, context):
    language = detect(query)

    if language == "bn":  # Bengali
        instruction = "উপরের প্রাসঙ্গিক তথ্য ব্যবহার করে প্রশ্নটির উত্তর বাংলায় দিন।"
    else:
        instruction = "Answer the question based on the above relevant information in English."

    prompt = f"""{instruction}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


# Main logic
def ask(query):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    index, chunks = load_faiss_index()
    query_vec = embed_query(query, model)
    top_chunks = get_top_chunks(query_vec, index, chunks)
    context = "\n".join(top_chunks)
    answer = generate_answer(query, context)
    return answer

# CLI for testing
if __name__ == "__main__":
    while True:
        user_q = input("\n🔍 Enter your query (Bangla/English): ")
        if user_q.lower() in ["exit", "quit"]:
            break
        try:
            result = ask(user_q)
            print(f"\n🤖 Answer: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
