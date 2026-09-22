import subprocess
import sys
import os


BASE_DIR = os.path.dirname(__file__)


def run_script(script_name):
    """Run another Python script from the data_pipeline folder."""

    script_path = os.path.join(
        BASE_DIR,
        script_name
    )

    print("\n" + "=" * 70)
    print(f"RUNNING: {script_name}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, script_path],
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with "
            f"exit code {result.returncode}"
        )


def main():

    print("=" * 70)
    print("ZEPTO DATA PIPELINE - END TO END")
    print("=" * 70)

    # Step 1: Scrape and clean
    run_script(
        "scrape_pipeline.py"
    )

    # Step 2: Create normalized SQLite database
    run_script(
        "database.py"
    )

    # Step 3: Run SQL queries and pandas comparison
    run_script(
        "sql_queries.py"
    )

    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
