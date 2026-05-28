"""
Task 1 - Amazon.in Laptop Scraper
Scrapes laptop listings: Image, Title, Rating, Price, Ad/Organic
Saves output to a timestamped CSV file.
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
import random
from datetime import datetime
import os


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
    "Referer": "https://www.amazon.in/",
}

BASE_URL = "https://www.amazon.in/s?k=laptops&page={page}"


def fetch_page(url: str, retries: int = 3) -> BeautifulSoup | None:
    """Fetch a page and return a BeautifulSoup object. Retries on failure."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                return BeautifulSoup(response.content, "html.parser")
            else:
                print(f"  [!] Status {response.status_code} on attempt {attempt + 1}")
        except requests.RequestException as e:
            print(f"  [!] Request error on attempt {attempt + 1}: {e}")
        time.sleep(random.uniform(2, 4))
    return None


def is_ad(item) -> str:
    """Check if a search result item is a sponsored (ad) listing."""
    # Amazon marks sponsored items with a label
    sponsored_label = item.find("span", string=lambda t: t and "Sponsored" in t)
    if sponsored_label:
        return "Ad"
    badge = item.find("span", {"class": lambda c: c and "puis-sponsored-label" in c})
    if badge:
        return "Ad"
    return "Organic"


def extract_product(item) -> dict | None:
    """Extract all required fields from a single search result item."""
    try:
        # Title
        title_tag = item.find("span", {"class": "a-size-medium"}) or \
                    item.find("span", {"class": "a-size-base-plus"}) or \
                    item.find("h2")
        title = title_tag.get_text(strip=True) if title_tag else "N/A"

        # Skip non-product items (e.g. banners)
        if title == "N/A" or len(title) < 5:
            return None

        # Price
        price_whole = item.find("span", {"class": "a-price-whole"})
        price = f"₹{price_whole.get_text(strip=True)}" if price_whole else "N/A"

        # Rating
        rating_tag = item.find("span", {"class": "a-icon-alt"})
        rating = rating_tag.get_text(strip=True).split(" ")[0] if rating_tag else "N/A"

        # Image URL
        img_tag = item.find("img", {"class": "s-image"})
        image_url = img_tag["src"] if img_tag and img_tag.get("src") else "N/A"

        # Ad or Organic
        result_type = is_ad(item)

        return {
            "Title": title,
            "Price": price,
            "Rating": rating,
            "Image URL": image_url,
            "Result Type": result_type,
        }

    except Exception as e:
        print(f"  [!] Error extracting product: {e}")
        return None


def scrape_laptops(pages: int = 5) -> list[dict]:
    """Scrape multiple pages of Amazon laptop search results."""
    all_products = []

    for page in range(1, pages + 1):
        url = BASE_URL.format(page=page)
        print(f"[*] Scraping page {page}: {url}")

        soup = fetch_page(url)
        if not soup:
            print(f"  [!] Could not fetch page {page}. Skipping.")
            continue

        # Each search result is inside this div
        items = soup.find_all("div", {"data-component-type": "s-search-result"})
        print(f"  [+] Found {len(items)} items on page {page}")

        for item in items:
            product = extract_product(item)
            if product:
                all_products.append(product)

        # Polite delay between pages
        delay = random.uniform(3, 6)
        print(f"  [~] Waiting {delay:.1f}s before next page...")
        time.sleep(delay)

    return all_products


def save_to_csv(products: list[dict], output_dir: str = ".") -> str:
    """Save the product list to a timestamped CSV file."""
    if not products:
        print("[!] No products to save.")
        return ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"amazon_laptops_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    fieldnames = ["Title", "Price", "Rating", "Image URL", "Result Type"]

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(products)

    print(f"\n[✓] Saved {len(products)} products to: {filepath}")
    return filepath


def main():
    print("=" * 60)
    print("   Amazon.in Laptop Scraper")
    print("=" * 60)

    products = scrape_laptops(pages=5)

    if products:
        save_to_csv(products, output_dir=".")
        print(f"\n[✓] Total products scraped: {len(products)}")

        # Quick summary
        ads = sum(1 for p in products if p["Result Type"] == "Ad")
        organic = len(products) - ads
        print(f"    Ads      : {ads}")
        print(f"    Organic  : {organic}")
    else:
        print("\n[!] No products were scraped. Amazon may be blocking requests.")
        print("    Try: rotating User-Agent, adding proxy, or using Selenium.")


if __name__ == "__main__":
    main()
