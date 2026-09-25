import os
import re
import statistics
import requests
import pandas as pd
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
FIXED_GBP_TO_INR = 105.50

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "cleaned_books.csv")


RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ZeptoDataPipeline/1.0)"
}


def get_soup(url):
    """Download a page and return its BeautifulSoup object."""

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def get_category_urls():
    """
    Get all available book categories from the main catalogue page.
    """

    soup = get_soup(BASE_URL)

    category_urls = {}

    for link in soup.select("div.side_categories ul li ul li a"):
        category_name = link.get_text(strip=True)
        href = link.get("href")

        if href:
            category_url = requests.compat.urljoin(
                BASE_URL,
                href
            )

            category_urls[category_name] = category_url

    return category_urls


def get_next_page_url(soup, current_url):
    """Return the URL of the next catalogue page."""

    next_link = soup.select_one("li.next a")

    if not next_link:
        return None

    href = next_link.get("href")

    if not href:
        return None

    return requests.compat.urljoin(
        current_url,
        href
    )


def parse_book_page(book_url, category_name):
    """
    Open an individual book page and extract the required fields.
    """

    soup = get_soup(book_url)

    title_element = soup.select_one("div.product_main h1")

    price_element = soup.select_one(
        "div.product_main p.price_color"
    )

    rating_element = soup.select_one(
        "div.product_main p.star-rating"
    )

    availability_element = soup.select_one(
        "div.product_main p.instock.availability"
    )

    title = (
        title_element.get_text(strip=True)
        if title_element
        else None
    )

    price = (
        price_element.get_text(strip=True)
        if price_element
        else None
    )

    star_rating = None

    if rating_element:
        rating_classes = rating_element.get("class", [])

        for rating_name in RATING_MAP:
            if rating_name in rating_classes:
                star_rating = rating_name
                break

    availability = (
        availability_element.get_text(" ", strip=True)
        if availability_element
        else None
    )

    return {
        "title": title,
        "price": price,
        "star_rating": star_rating,
        "availability": availability,
        "category": category_name,
    }


def scrape_category(category_name, category_url, max_books=25):
    """
    Scrape books from one category.

    max_books limits the number of books collected from each category.
    """

    books = []
    current_url = category_url

    while current_url and len(books) < max_books:

        print(
            f"Scraping category='{category_name}' "
            f"page='{current_url}'"
        )

        soup = get_soup(current_url)

        book_links = soup.select(
            "article.product_pod h3 a"
        )

        for book_link in book_links:

            if len(books) >= max_books:
                break

            href = book_link.get("href")

            if not href:
                continue

            book_url = requests.compat.urljoin(
                current_url,
                href
            )

            try:
                book_data = parse_book_page(
                    book_url,
                    category_name
                )

                books.append(book_data)

            except requests.RequestException as error:
                print(
                    f"Skipping book because request failed: {error}"
                )

            except Exception as error:
                print(
                    f"Skipping book because parsing failed: {error}"
                )

        current_url = get_next_page_url(
            soup,
            current_url
        )

    return books


def clean_price(value):
    """
    Convert a GBP price such as '£51.77' to float 51.77.
    """

    if value is None:
        return None

    try:
        cleaned = re.sub(
            r"[^0-9.]",
            "",
            str(value)
        )

        if cleaned == "":
            return None

        return float(cleaned)

    except (ValueError, TypeError):
        return None


def clean_rating(value):
    """
    Convert textual rating One...Five into an integer.
    """

    if value in RATING_MAP:
        return RATING_MAP[value]

    return None


def clean_stock(value):
    """
    Convert availability text into boolean.
    """

    if value is None:
        return None

    text = str(value).strip().lower()

    if "in stock" in text:
        return True

    if "out of stock" in text:
        return False

    return None


def clean_dataframe(df):
    """
    Clean all scraped fields and handle parsing failures.
    """

    df = df.copy()

    df["price_gbp"] = df["price"].apply(clean_price)

    df["rating"] = df["star_rating"].apply(clean_rating)

    df["in_stock"] = df["availability"].apply(clean_stock)

    # Numeric fields:
    # Invalid price/rating values are imputed using their median.
    for column in ["price_gbp", "rating"]:

        if df[column].isna().any():

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

    # Availability is required to be boolean.
    # If availability cannot be parsed, drop that row.
    df = df.dropna(
        subset=["in_stock"]
    )

    df["price_gbp"] = df["price_gbp"].astype(float)

    df["rating"] = df["rating"].round().astype(int)

    df["in_stock"] = df["in_stock"].astype(bool)

    # Required fixed project conversion.
    df["price_inr"] = (
        df["price_gbp"] * FIXED_GBP_TO_INR
    ).round(2)

    # Keep only the required final fields.
    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ]

    return df


def main():

    print("=" * 70)
    print("ZEPTO DATA PIPELINE")
    print("=" * 70)

    print(
        f"\nFixed currency conversion: "
        f"1 GBP = {FIXED_GBP_TO_INR} INR"
    )

    print("\nGetting book categories...")

    categories = get_category_urls()

    print(
        f"Found {len(categories)} categories."
    )

    # Select at least three categories.
    # More categories are selected when available.
    selected_categories = list(
        categories.items()
    )[:4]

    print("\nSelected categories:")

    for category_name, category_url in selected_categories:
        print(
            f" - {category_name}: {category_url}"
        )

    all_books = []

    for category_name, category_url in selected_categories:

        category_books = scrape_category(
            category_name,
            category_url,
            max_books=25
        )

        print(
            f"Collected {len(category_books)} books "
            f"from {category_name}"
        )

        all_books.extend(category_books)

    print(
        f"\nTotal raw books collected: "
        f"{len(all_books)}"
    )

    if len(all_books) < 60:

        raise RuntimeError(
            "The pipeline collected fewer than "
            "60 books. Please increase the number "
            "of categories/pages scraped."
        )

    raw_df = pd.DataFrame(all_books)

    print("\nRaw data preview:")
    print(raw_df.head())

    print("\nCleaning data...")

    clean_df = clean_dataframe(raw_df)

    print(
        f"Cleaned dataset contains "
        f"{len(clean_df)} rows."
    )

    if len(clean_df) < 60:

        raise RuntimeError(
            "Fewer than 60 valid rows remain "
            "after cleaning."
        )

    print("\nFinal column types:")

    print(
        clean_df.dtypes
    )

    print("\nFinal data preview:")

    print(
        clean_df.head(10).to_string(index=False)
    )

    clean_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\nCleaned dataset saved to:\n"
        f"{OUTPUT_CSV}"
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
