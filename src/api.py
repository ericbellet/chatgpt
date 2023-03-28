from fastapi import FastAPI
from pydantic import BaseModel

class Question(BaseModel):
    question_text: str

from src.gpt import GPT
chat = GPT()
vectorDB = chat.load_vectorDB()

app = FastAPI()

@app.post("/ask")
async def ask_question(question: Question):
    result = chat.query(question.question_text)
    return {"answer": result}
