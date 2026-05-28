"""
Task 2A - Face Authentication: Testing / Prediction
====================================================
Standalone function to load the model and predict
whether two face images belong to the same person.

Usage:
    python test_predict.py --img1 path/to/face1.jpg --img2 path/to/face2.jpg

Or import and call predict() from your own code.
"""

import argparse
import numpy as np
import cv2
from deepface import DeepFace


# Cosine similarity threshold — tune based on your use case
SIMILARITY_THRESHOLD = 0.68


# ─────────────────────────────────────────────
# Core prediction function
# ─────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img


def get_embedding(image: np.ndarray, model_name: str = "Facenet") -> np.ndarray:
    """
    Extract face embedding using DeepFace (pretrained Facenet).

    Args:
        image      : BGR numpy array
        model_name : DeepFace model to use ('Facenet', 'VGG-Face', etc.)

    Returns:
        1-D numpy array — the face embedding vector
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = DeepFace.represent(
        img_path=rgb_image,
        model_name=model_name,
        enforce_detection=False,
        detector_backend="opencv",
    )
    return np.array(result[0]["embedding"])


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (n1 * n2))


def detect_faces(image: np.ndarray) -> list[dict]:
    """Return bounding boxes of detected faces using OpenCV Haar Cascade."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return []
    return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for x, y, w, h in faces]


def predict(image1_path: str, image2_path: str, model_name: str = "Facenet") -> dict:
    """
    Compare two face images and return the verification result.

    Args:
        image1_path : Path to the first face image
        image2_path : Path to the second face image
        model_name  : DeepFace model name (default: 'Facenet')

    Returns:
        dict with keys:
            - verification_result : 'same person' or 'different person'
            - similarity_score    : float in [0, 1]
            - bounding_boxes      : dict with 'image1' and 'image2' lists
    """
    # Load images
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    # Detect faces
    boxes1 = detect_faces(img1)
    boxes2 = detect_faces(img2)

    # Extract embeddings
    emb1 = get_embedding(img1, model_name)
    emb2 = get_embedding(img2, model_name)

    # Compute similarity
    raw_similarity = cosine_similarity(emb1, emb2)
    # Map from [-1, 1] to [0, 1]
    similarity = max(0.0, min(1.0, (raw_similarity + 1) / 2))

    result = "same person" if similarity >= SIMILARITY_THRESHOLD else "different person"

    return {
        "verification_result": result,
        "similarity_score": round(similarity, 4),
        "threshold_used": SIMILARITY_THRESHOLD,
        "bounding_boxes": {
            "image1": boxes1,
            "image2": boxes2,
        },
    }


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Authentication — Prediction")
    parser.add_argument("--img1", required=True, help="Path to first face image")
    parser.add_argument("--img2", required=True, help="Path to second face image")
    parser.add_argument(
        "--model",
        default="Facenet",
        choices=["Facenet", "VGG-Face", "ArcFace", "Dlib"],
        help="Embedding model to use (default: Facenet)",
    )
    args = parser.parse_args()

    print(f"\n[*] Comparing:\n    Image 1: {args.img1}\n    Image 2: {args.img2}")
    output = predict(args.img1, args.img2, model_name=args.model)

    print("\n──────────────────────────────────")
    print(f"  Result     : {output['verification_result'].upper()}")
    print(f"  Similarity : {output['similarity_score']} (threshold: {output['threshold_used']})")
    print(f"  Faces in image1: {len(output['bounding_boxes']['image1'])}")
    print(f"  Faces in image2: {len(output['bounding_boxes']['image2'])}")
    print("──────────────────────────────────\n")
