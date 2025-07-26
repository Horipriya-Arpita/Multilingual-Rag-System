from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.rag import ask

app = FastAPI()

class QueryInput(BaseModel):
    query: str

@app.post("/ask")
async def get_answer(data: QueryInput):
    answer = ask(data.query)
    return {"query": data.query, "answer": answer}
