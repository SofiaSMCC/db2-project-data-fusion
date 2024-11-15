from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from main import querySearch, setupIndex
import regex as re
import psycopg2
import uvicorn

class Base(BaseModel):
    query: str

app = FastAPI()
connection = psycopg2.connect(database="postgres", user="postgres", password="docker", host="localhost", port=5432)
cursor = connection.cursor()

@app.post("/query")
async def root(query: Base):
    if query.query == "":
        return []

    return parseQuery(query.query)

def parseQuery(query: str):
    q = [p.lower() for p in re.split("( |\\\".*?\\\"|'.*?')", query) if p.strip()]
    res = []

    if q[0] == "select":
        if q[3] == "songslyrics" and q[5] == "lyric":
            if q[8] == "selfindex":
                match = q[6]
                k = int(q[10])

                res = querySearch(match, top_k=k)
                print(res)
                if(res == None):
                    return []
            elif q[8] == "spimi":
                cursor.execute("SELECT * FROM dataset")
                record = cursor.fetchall()

                print(record)

    return res

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    setupIndex()
    uvicorn.run(app)