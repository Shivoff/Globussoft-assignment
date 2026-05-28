# Globussoft Assignment

## Task 1 - Amazon Scraper
Scrapes laptop data from amazon.in and saves to CSV file with timestamp.

## Task 2 - Face Authentication
FastAPI service to verify if two face images are the same person.

## How to run
pip install -r requirements.txt
python amazon_scraper.py
uvicorn app:app --reload

## Files
- amazon_scraper.py - Task 1 scraper
- app.py - FastAPI app
- train_model.py - Model setup
- test_predict.py - Testing script
