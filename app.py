"""
Task 2A - Face Authentication Service
FastAPI service that:
  - Accepts two face images
  - Detects faces using OpenCV Haar Cascade
  - Extracts embeddings using DeepFace (FaceNet / VGG-Face)
  - Computes cosine similarity
  - Returns: verification result, similarity score, bounding boxes
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from PIL import Image
from deepface import DeepFace
import base64


app = FastAPI(
    title="Face Authentication API",
    description="Verify whether two face images belong to the same person.",
    version="1.0.0",
)

# Load OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Similarity threshold — faces with cosine similarity >= this are "same person"
SIMILARITY_THRESHOLD = 0.68


def read_image_from_upload(upload_file: UploadFile) -> np.ndarray:
    """Read an uploaded file and convert it to an OpenCV BGR image."""
    contents = upload_file.file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv_image


def detect_faces(image: np.ndarray) -> list[dict]:
    """
    Detect faces in an image using OpenCV Haar Cascade.
    Returns a list of bounding boxes: [{"x", "y", "w", "h"}]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return []
    return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for x, y, w, h in faces]


def get_embedding(image: np.ndarray) -> np.ndarray:
    """
    Extract a face embedding vector using DeepFace (Facenet model).
    Returns a 1-D numpy array.
    """
    # DeepFace expects an RGB image or a file path
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = DeepFace.represent(
        img_path=rgb_image,
        model_name="Facenet",
        enforce_detection=False,   # Don't crash if no face; we already checked
        detector_backend="opencv",
    )
    embedding = np.array(result[0]["embedding"])
    return embedding


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors. Returns value in [-1, 1]."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


@app.get("/")
def root():
    return {"message": "Face Authentication API is running. POST to /verify"}


@app.post("/verify")
async def verify_faces(
    image1: UploadFile = File(..., description="First face image"),
    image2: UploadFile = File(..., description="Second face image"),
):
    """
    Compare two face images and return:
    - verification_result: 'same person' or 'different person'
    - similarity_score: cosine similarity (0 to 1)
    - bounding_boxes: detected face locations in each image
    """

    # --- Read images ---
    try:
        img1 = read_image_from_upload(image1)
        img2 = read_image_from_upload(image2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image(s): {e}")

    # --- Detect faces ---
    boxes1 = detect_faces(img1)
    boxes2 = detect_faces(img2)

    if not boxes1:
        raise HTTPException(status_code=422, detail="No face detected in image 1.")
    if not boxes2:
        raise HTTPException(status_code=422, detail="No face detected in image 2.")

    # --- Extract embeddings ---
    try:
        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {e}")

    # --- Compute similarity ---
    similarity = cosine_similarity(emb1, emb2)
    # Clamp to [0, 1] range for cleaner output
    similarity_clamped = max(0.0, min(1.0, (similarity + 1) / 2))

    verification_result = (
        "same person" if similarity_clamped >= SIMILARITY_THRESHOLD else "different person"
    )

    return JSONResponse(
        content={
            "verification_result": verification_result,
            "similarity_score": round(similarity_clamped, 4),
            "threshold_used": SIMILARITY_THRESHOLD,
            "bounding_boxes": {
                "image1": boxes1,
                "image2": boxes2,
            },
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
