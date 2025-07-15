import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles # 👈 추가

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

app.mount("/img", StaticFiles(directory=os.path.join(BASE_DIR, "img")), name="img")
app.mount("/tmp", StaticFiles(directory=os.path.join(BASE_DIR, "tmp")), name="tmp")

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(BASE_DIR, 'index.html'))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)