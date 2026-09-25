import os
import sqlite3
import pandas as pd


BASE_DIR = os.path.dirname(__file__)

CSV_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "cleaned_books.csv"
)

DB_PATH = os.path.join(
    BASE_DIR,
    "zepto_books.db"
)


def create_database():
    """Create the normalized SQLite database."""

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Cleaned CSV not found: {CSV_PATH}\n"
            "Run scrape_pipeline.py first."
        )

    # Read cleaned data
    df = pd.read_csv(CSV_PATH)

    print("=" * 70)
    print("CREATING ZEPTO BOOK DATABASE")
    print("=" * 70)

    print(f"\nInput CSV: {CSV_PATH}")
    print(f"Rows loaded: {len(df)}")

    # ---------------------------------------------------------
    # Connect to SQLite
    # ---------------------------------------------------------

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    # Enable foreign-key enforcement
    cursor.execute("PRAGMA foreign_keys = ON")

    # ---------------------------------------------------------
    # Drop existing tables
    # ---------------------------------------------------------

    cursor.execute(
        "DROP TABLE IF EXISTS books"
    )

    cursor.execute(
        "DROP TABLE IF EXISTS categories"
    )

    # ---------------------------------------------------------
    # Create categories table
    # ---------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
        """
    )

    # ---------------------------------------------------------
    # Create books table
    # ---------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    # ---------------------------------------------------------
    # Insert categories
    # ---------------------------------------------------------

    categories = sorted(
        df["category"].dropna().unique()
    )

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories (category_name)
            VALUES (?)
            """,
            (category,)
        )

    # ---------------------------------------------------------
    # Create category lookup
    # ---------------------------------------------------------

    cursor.execute(
        """
        SELECT category_id, category_name
        FROM categories
        """
    )

    category_lookup = {
        category_name: category_id
        for category_id, category_name in cursor.fetchall()
    }

    # ---------------------------------------------------------
    # Insert books
    # ---------------------------------------------------------

    for _, row in df.iterrows():

        category_id = category_lookup[
            row["category"]
        ]

        cursor.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(bool(row["in_stock"])),
                category_id
            )
        )

    # ---------------------------------------------------------
    # Commit changes
    # ---------------------------------------------------------

    connection.commit()

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------

    cursor.execute(
        "SELECT COUNT(*) FROM books"
    )

    book_count = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM categories"
    )

    category_count = cursor.fetchone()[0]

    print("\nDatabase created successfully.")

    print(
        f"Books inserted: {book_count}"
    )

    print(
        f"Categories inserted: {category_count}"
    )

    print(
        f"Database location: {DB_PATH}"
    )

    # Show sample records from JOIN
    cursor.execute(
        """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        LIMIT 5
        """
    )

    sample_rows = cursor.fetchall()

    print("\nSample JOIN result:")

    for row in sample_rows:
        print(row)

    connection.close()


if __name__ == "__main__":
    create_database()
