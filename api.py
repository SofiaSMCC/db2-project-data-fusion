from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import buscar_letra
import regex as re

class Base(BaseModel):
    query: str

app = FastAPI()

@app.post("/query")
async def root(query: Base):
    if query.query == "":
        return []

    return parseQuery(query.query)

def parseQuery(query: str):
    q = [p.lower() for p in re.split("( |\\\".*?\\\"|'.*?')", query) if p.strip()]

    if q[0] == "select" and q[5] == "lyric" and q[7] == "limit":
        match = q[6]
        k = int(q[8])

        res = buscar_letra(match, top_k=k)
        if(res == None):
            return []

        return res

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)