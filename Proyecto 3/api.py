from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
from Experiments.RTree_exp import Run_KnnRtree
from Experiments.HighD_exp import Run_KnnLSH
from Experiments.Sequential_exp import Run_KnnSequential, Run_RangeSearch

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Crear directorio de uploads si no existe

BASE_DIR = os.path.dirname(os.path.abspath(_file_))
IMAGES_DIR = os.path.join(BASE_DIR, "poke2")

@app.post("/knn-sequential")
async def knn_sequential(file: UploadFile, k: int = Form(...)):
    # Guardar archivo
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Ejecutar búsqueda KNN sin indexación
    results = Run_KnnSequential(file_path, k=k)
    return JSONResponse(content={"algorithm": "knn-sequential", "results": results})

@app.post("/range-search")
async def range_search(file: UploadFile, radius: float = Form(...)):
    # Guardar archivo
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Ejecutar búsqueda por rango
    results = Run_RangeSearch(file_path, radius=radius)
    return JSONResponse(content={"algorithm": "range-search", "results": results})

@app.post("/knn-rtree")
async def knn_rtree(file: UploadFile, k: int = Form(...)):
    # Guardar archivo
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Ejecutar búsqueda KNN con R-tree
    results = Run_KnnRtree(file_path, k=k)
    return JSONResponse(content={"algorithm": "knn-rtree", "results": results})

@app.post("/knn-faiss")
async def knn_faiss(file: UploadFile, k: int = Form(...)):
    # Guardar archivo
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Ejecutar búsqueda KNN con FAISS
    results = Run_KnnLSH(file_path, k=k)
    return JSONResponse(content={"algorithm": "knn-faiss", "results": results})

@app.get("/poke2")
async def get_image(img: str):

    # Construir la ruta relativa y luego la ruta absoluta
    relative_path = os.path.join("poke2", img)
    full_path = os.path.join(BASE_DIR, relative_path)

    print(full_path)

    # Verificar si la imagen existe
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Retornar la imagen como archivo
    return FileResponse(full_path)

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
