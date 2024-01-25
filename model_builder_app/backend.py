import webbrowser

from fastapi import FastAPI

app = FastAPI()


@app.post("/model_builder/")
async def return_message():
    return {"message": "Please access `http://localhost:8002` in your browser"}