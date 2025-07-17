from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from deepface import DeepFace
import tempfile
import shutil
import os

app = FastAPI()

@app.get("/")
@app.get("/hello")
def say_hello():
    return {"message": "Hello, world!"}

@app.post("/detect-emotion/")
async def detect_emotion(file: UploadFile = File(...)):
    try:
        # Save the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # Run DeepFace emotion analysis
        analysis = DeepFace.analyze(
            img_path=temp_path,
            actions=["emotion"],
            enforce_detection=False
        )

        emotion_scores = analysis[0]["emotion"]
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)

        # Clean up the temporary file
        os.remove(temp_path)

        return dominant_emotion

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
