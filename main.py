from fastapi import FastAPI,Query
from rag_pipeline import load_pipeline, ask_question
 
 
app = FastAPI(title = "Ecommerce RAG API")
chain = load_pipeline()
 

 
@app.post("/chat")
def chat(query: str = Query(..., description = "User query")):
    result = ask_question(chain, query)
    return result