# Globussoft — Data Science Assignment

Submitted for the **Junior Data Science Role**.

---

## Project Structure

```
globussoft/
├── task1/
│   └── amazon_scraper.py        # Amazon.in laptop scraper
├── task2/
│   ├── app.py                   # FastAPI face authentication service
│   ├── train_model.py           # Model download + preparation
│   └── test_predict.py          # Standalone prediction function
├── sample_images/
│   ├── face1.jpg                # Sample face image 1
│   └── face2.jpg                # Sample face image 2
├── requirements.txt
└── README.md
```

---

## Task 1 — Amazon Laptop Scraper

Scrapes Amazon.in for laptop search results and saves the data to a **timestamped CSV file**.

**Fields collected:**
- Title
- Price
- Rating
- Image URL
- Result Type (Ad / Organic)

### Run

```bash
cd task1
python amazon_scraper.py
```

Output: `amazon_laptops_YYYYMMDD_HHMMSS.csv` in the current directory.

**Note:** Amazon actively blocks automated scraping. If you see empty results,
try adding a proxy or increasing the request delay inside the script.

---

## Task 2 — Face Authentication (Option A)

A **FastAPI** service that compares two face images and decides whether
they belong to the same person.

**Stack:**
- OpenCV Haar Cascade — face detection
- DeepFace + Facenet — face embedding extraction
- Cosine similarity — comparison metric

**Returns:**
- `verification_result` — `"same person"` or `"different person"`
- `similarity_score` — float in [0, 1]
- `bounding_boxes` — face locations in each image

---

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2 — Prepare the model (download weights)

```bash
cd task2
python train_model.py
```

This downloads the pretrained Facenet weights (~90 MB) into `~/.deepface/weights/`.
Only needs to run once.

---

### Step 3 — Start the API server

```bash
cd task2
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Step 4 — Test via command line

```bash
python test_predict.py --img1 ../sample_images/face1.jpg --img2 ../sample_images/face2.jpg
```

---

### Step 5 — Test via curl

```bash
curl -X POST "http://localhost:8000/verify" \
     -F "image1=@../sample_images/face1.jpg" \
     -F "image2=@../sample_images/face2.jpg"
```

Example response:

```json
{
  "verification_result": "same person",
  "similarity_score": 0.8923,
  "threshold_used": 0.68,
  "bounding_boxes": {
    "image1": [{"x": 120, "y": 80, "w": 200, "h": 200}],
    "image2": [{"x": 95,  "y": 70, "w": 190, "h": 195}]
  }
}
```

---

## Notes

- Python 3.10+ recommended
- CPU inference works fine; GPU speeds up embedding extraction
- Similarity threshold (default `0.68`) can be adjusted in `app.py` and `test_predict.py`
