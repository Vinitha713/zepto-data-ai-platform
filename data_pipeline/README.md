# Data Pipeline

This module implements an end-to-end data engineering pipeline for Zepto.

## Pipeline

1. Scrape book data from Books to Scrape.
2. Clean and validate the scraped data.
3. Convert GBP prices to INR using the required fixed rate.
4. Store the cleaned data in a normalized SQLite database.
5. Execute SQL queries.
6. Read query results into pandas.
7. Reproduce the JOIN result using pandas merge.

## Currency Conversion

The project-defined fixed conversion rate is:

**1 GBP = 105.50 INR**

No external currency API is required.

## Data Source

https://books.toscrape.com/
