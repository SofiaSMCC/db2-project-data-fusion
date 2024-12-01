from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from inverted_index import InvertedIndex
from postgres_query import PostgresQuery
from pydantic import BaseModel
import regex as re
import uvicorn

app = FastAPI()
index = InvertedIndex('Proyecto 2/utils/dataset.csv')
postgres = PostgresQuery()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def root(req: QueryRequest):
    print(req.query)
    if req.query == "":
        return []

    return parseQuery(req.query)

def parseQuery(query: str):
    q = [p.lower() for p in re.split("( |\\\".*?\\\"|'.*?')", query) if p.strip()]
    res = []
    
    operator = q[0]
    condition = q[3]
    text_query = q[5]
    type = q[7]
    limit = q[9]

    if operator == "select":
        if condition == "lyrics":
            if type == "spimi":
                res = index.query_search(text_query, top_k=int(limit))
                if(res == None):
                    return []
                
            elif type == "postgresql":
                res = postgres.query_search(text_query, top_k=int(limit))
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
    index.spimi_invert()
    postgres.create_table()
    uvicorn.run(app)