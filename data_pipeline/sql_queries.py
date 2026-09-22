import os
import sqlite3
import pandas as pd


BASE_DIR = os.path.dirname(__file__)

DB_PATH = os.path.join(
    BASE_DIR,
    "zepto_books.db"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "sql_outputs.txt"
)


def run_query(connection, title, query):
    """Execute a SQL query and return a pandas DataFrame."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("\nSQL:")
    print(query)

    result = pd.read_sql(
        query,
        connection
    )

    print("\nResult:")
    print(
        result.to_string(index=False)
    )

    return result


def main():

    if not os.path.exists(DB_PATH):

        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Run database.py first."
        )

    connection = sqlite3.connect(
        DB_PATH
    )

    output_sections = []

    # =========================================================
    # Query 1: SELECT
    # =========================================================

    query_1 = """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock
        FROM books
    """

    result_1 = run_query(
        connection,
        "QUERY 1 - SELECT",
        query_1
    )

    output_sections.append(
        "QUERY 1 - SELECT\n"
        + result_1.head(10).to_string(index=False)
    )

    # =========================================================
    # Query 2: WHERE
    # =========================================================

    query_2 = """
        SELECT
            title,
            price_gbp,
            rating,
            in_stock
        FROM books
        WHERE rating >= 4
    """

    result_2 = run_query(
        connection,
        "QUERY 2 - WHERE",
        query_2
    )

    output_sections.append(
        "QUERY 2 - WHERE\n"
        + result_2.to_string(index=False)
    )

    # =========================================================
    # Query 3: ORDER BY + LIMIT
    # =========================================================

    query_3 = """
        SELECT
            title,
            price_gbp,
            price_inr
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10
    """

    result_3 = run_query(
        connection,
        "QUERY 3 - ORDER BY + LIMIT",
        query_3
    )

    output_sections.append(
        "QUERY 3 - ORDER BY + LIMIT\n"
        + result_3.to_string(index=False)
    )

    # =========================================================
    # Query 4: DISTINCT
    # =========================================================

    query_4 = """
        SELECT DISTINCT
            rating
        FROM books
        ORDER BY rating
    """

    result_4 = run_query(
        connection,
        "QUERY 4 - DISTINCT",
        query_4
    )

    output_sections.append(
        "QUERY 4 - DISTINCT\n"
        + result_4.to_string(index=False)
    )

    # =========================================================
    # Query 5: BETWEEN
    # =========================================================

    query_5 = """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp
    """

    result_5 = run_query(
        connection,
        "QUERY 5 - BETWEEN",
        query_5
    )

    output_sections.append(
        "QUERY 5 - BETWEEN\n"
        + result_5.to_string(index=False)
    )

    # =========================================================
    # Query 6: IN
    # =========================================================

    query_6 = """
        SELECT
            title,
            rating,
            in_stock
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC
    """

    result_6 = run_query(
        connection,
        "QUERY 6 - IN",
        query_6
    )

    output_sections.append(
        "QUERY 6 - IN\n"
        + result_6.to_string(index=False)
    )

    # =========================================================
    # Query 7: JOIN
    # =========================================================

    query_7 = """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name
        FROM books AS b
        INNER JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY c.category_name, b.title
    """

    sql_join_result = run_query(
        connection,
        "QUERY 7 - JOIN",
        query_7
    )

    output_sections.append(
        "QUERY 7 - JOIN\n"
        + sql_join_result.to_string(index=False)
    )

    # =========================================================
    # Pandas read_sql JOIN
    # =========================================================

    pandas_read_sql_result = pd.read_sql(
        query_7,
        connection
    )

    print("\n" + "=" * 70)
    print("PANDAS read_sql JOIN RESULT")
    print("=" * 70)

    print(
        pandas_read_sql_result.head(10).to_string(
            index=False
        )
    )

    # =========================================================
    # Pandas merge JOIN
    # =========================================================

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        connection
    )

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection
    )

    pandas_merge_result = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    pandas_merge_result = pandas_merge_result[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name"
        ]
    ].sort_values(
        ["category_name", "title"]
    ).reset_index(
        drop=True
    )

    pandas_read_sql_sorted = (
        pandas_read_sql_result[
            [
                "book_id",
                "title",
                "price_gbp",
                "price_inr",
                "rating",
                "in_stock",
                "category_name"
            ]
        ]
        .sort_values(
            ["category_name", "title"]
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("PANDAS merge JOIN RESULT")
    print("=" * 70)

    print(
        pandas_merge_result.head(10).to_string(
            index=False
        )
    )

    # =========================================================
    # Compare SQL JOIN and pandas merge
    # =========================================================

    join_results_match = (
        pandas_read_sql_sorted.equals(
            pandas_merge_result
        )
    )

    print("\n" + "=" * 70)
    print("JOIN COMPARISON")
    print("=" * 70)

    print(
        f"pd.read_sql() and pd.merge() "
        f"results match: {join_results_match}"
    )

    output_sections.append(
        "\nJOIN COMPARISON\n"
        f"pd.read_sql() and pd.merge() "
        f"results match: {join_results_match}"
    )

    # =========================================================
    # Save query outputs
    # =========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n\n".join(output_sections)
        )

    print(
        f"\nSQL output saved to:\n{OUTPUT_FILE}"
    )

    connection.close()

    print(
        "\nAll SQL queries completed successfully."
    )


if __name__ == "__main__":
    main()
