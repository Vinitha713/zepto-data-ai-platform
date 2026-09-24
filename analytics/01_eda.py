import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(__file__)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

CSV_FILE = os.path.join(
    BASE_DIR,
    "titanic.csv"
)

REPORT_FILE = os.path.join(
    BASE_DIR,
    "eda_report.md"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def iqr_outlier_count(series):
    """Return number of IQR-based outliers."""

    series = series.dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return int(
        ((series < lower) | (series > upper)).sum()
    )


def save_plot(filename):
    """Save chart into analytics/outputs."""

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 1. LOAD TITANIC DATASET ONCE
# ============================================================

print("=" * 70)
print("LOADING TITANIC DATASET")
print("=" * 70)

df = sns.load_dataset(
    "titanic"
)

print("\nDATASET INFO")
print("-" * 70)

df.info()

print("\nDATASET DESCRIPTION")
print("-" * 70)

print(
    df.describe(
        include="all"
    )
)

print("\nDATASET SHAPE")
print("-" * 70)

print(
    df.shape
)


# ============================================================
# SAVE RAW OFFLINE FALLBACK
# ============================================================

df.to_csv(
    CSV_FILE,
    index=False
)

print(
    f"\nRaw offline fallback saved to: {CSV_FILE}"
)


# ============================================================
# 2. MISSING VALUE ANALYSIS
# ============================================================

missing_counts = df.isnull().sum()

missing_percentages = (
    df.isnull().mean() * 100
)

missing_table = pd.DataFrame(
    {
        "missing_count": missing_counts,
        "missing_percentage": missing_percentages
    }
)

missing_table = missing_table[
    missing_table["missing_count"] > 0
].sort_values(
    "missing_percentage",
    ascending=False
)

print("\nMISSING VALUES")
print("-" * 70)

print(
    missing_table
)


# ============================================================
# 3. CLEANING
# ============================================================

cleaned_df = df.copy()

cleaning_notes = []


for column in missing_table.index:

    missing_pct = missing_percentages[column]

    # Less than 5% -> drop affected rows
    if missing_pct < 5:

        before = len(cleaned_df)

        cleaned_df = cleaned_df.dropna(
            subset=[column]
        )

        removed = before - len(cleaned_df)

        cleaning_notes.append(
            f"- `{column}` had "
            f"{missing_pct:.2f}% missing values. "
            f"Because this is below 5%, rows missing "
            f"`{column}` were dropped "
            f"({removed} rows removed)."
        )

    # 5% to 30% -> impute
    elif missing_pct <= 30:

        if pd.api.types.is_numeric_dtype(
            cleaned_df[column]
        ):

            median_value = cleaned_df[
                column
            ].median()

            cleaned_df[column] = (
                cleaned_df[column].fillna(
                    median_value
                )
            )

            cleaning_notes.append(
                f"- `{column}` had "
                f"{missing_pct:.2f}% missing values. "
                f"Because this is between 5% and 30%, "
                f"missing numeric values were imputed "
                f"with the median "
                f"({median_value:.3f})."
            )

        else:

            mode_value = cleaned_df[
                column
            ].mode()[0]

            cleaned_df[column] = (
                cleaned_df[column].fillna(
                    mode_value
                )
            )

            cleaning_notes.append(
                f"- `{column}` had "
                f"{missing_pct:.2f}% missing values. "
                f"Because this is between 5% and 30%, "
                f"missing categorical values were imputed "
                f"with the mode `{mode_value}`."
            )

    # More than 30% -> drop column
    else:

        cleaned_df = cleaned_df.drop(
            columns=[column]
        )

        cleaning_notes.append(
            f"- `{column}` had "
            f"{missing_pct:.2f}% missing values. "
            f"Because this exceeds 30%, the column "
            f"was dropped because imputation would be "
            f"unreliable."
        )


# Remove rows that may still contain missing values
cleaned_df = cleaned_df.dropna(
    axis=0
).reset_index(
    drop=True
)


# ============================================================
# SAVE CLEANED DATA
# ============================================================

cleaned_df.to_csv(
    CSV_FILE,
    index=False
)

print("\nCLEANED DATASET")
print("-" * 70)

print(
    cleaned_df.shape
)

print(
    f"\nCleaned dataset saved to: {CSV_FILE}"
)


# ============================================================
# 4. UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("UNIVARIATE ANALYSIS")
print("=" * 70)


# -----------------------------
# AGE HISTOGRAM
# -----------------------------

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    cleaned_df["age"],
    kde=True
)

plt.title(
    "Age Distribution"
)

plt.xlabel(
    "Age"
)

plt.ylabel(
    "Count"
)

save_plot(
    "01_age_histogram.png"
)


# -----------------------------
# AGE BOX PLOT
# -----------------------------

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    x=cleaned_df["age"]
)

plt.title(
    "Age Box Plot"
)

save_plot(
    "02_age_boxplot.png"
)


# -----------------------------
# FARE HISTOGRAM
# -----------------------------

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    cleaned_df["fare"],
    kde=True
)

plt.title(
    "Fare Distribution"
)

plt.xlabel(
    "Fare"
)

plt.ylabel(
    "Count"
)

save_plot(
    "03_fare_histogram.png"
)


# -----------------------------
# FARE BOX PLOT
# -----------------------------

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    x=cleaned_df["fare"]
)

plt.title(
    "Fare Box Plot"
)

save_plot(
    "04_fare_boxplot.png"
)


# ============================================================
# IQR OUTLIERS
# ============================================================

age_outliers = iqr_outlier_count(
    cleaned_df["age"]
)

fare_outliers = iqr_outlier_count(
    cleaned_df["fare"]
)

print(
    f"\nAge IQR outliers: {age_outliers}"
)

print(
    f"Fare IQR outliers: {fare_outliers}"
)


# ============================================================
# FARE STATISTICS
# ============================================================

fare_mean = cleaned_df[
    "fare"
].mean()

fare_median = cleaned_df[
    "fare"
].median()

fare_mode = cleaned_df[
    "fare"
].mode()[0]


if (
    fare_mean > fare_median
    and fare_median > fare_mode
):

    fare_skew_text = (
        "Fare is right-skewed because "
        "the mean is greater than the median, "
        "and the median is greater than the mode."
    )

elif (
    fare_mean < fare_median
    and fare_median < fare_mode
):

    fare_skew_text = (
        "Fare is left-skewed because "
        "the mean is less than the median, "
        "and the median is less than the mode."
    )

else:

    fare_skew_text = (
        "Fare does not follow a simple "
        "mean > median > mode or "
        "mean < median < mode ordering; "
        "the distribution should be interpreted "
        "using the histogram as well."
    )


# ============================================================
# 5. BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("BIVARIATE ANALYSIS")
print("=" * 70)


# -----------------------------
# Survival by sex
# -----------------------------

male_survival = cleaned_df.loc[
    cleaned_df["sex"] == "male",
    "survived"
].mean()

female_survival = cleaned_df.loc[
    cleaned_df["sex"] == "female",
    "survived"
].mean()


# -----------------------------
# Survival by class
# -----------------------------

class_survival = (
    cleaned_df
    .groupby("pclass")["survived"]
    .mean()
)


# -----------------------------
# Survival by sex AND class
# -----------------------------

sex_class_survival = (
    cleaned_df
    .groupby(
        ["sex", "pclass"]
    )["survived"]
    .mean()
)


print("\nSurvival rate by sex:")

print(
    pd.Series(
        {
            "female": female_survival,
            "male": male_survival
        }
    )
)


print("\nSurvival rate by passenger class:")

print(
    class_survival
)


print(
    "\nSurvival rate by sex and passenger class:"
)

print(
    sex_class_survival
)


# ============================================================
# BOOLEAN MASKING EXAMPLE
# ============================================================

female_first_class = cleaned_df.loc[
    (
        (cleaned_df["sex"] == "female")
        &
        (cleaned_df["pclass"] == 1)
    ),
    "survived"
].mean()

male_third_class = cleaned_df.loc[
    (
        (cleaned_df["sex"] == "male")
        &
        (cleaned_df["pclass"] == 3)
    ),
    "survived"
].mean()


# ============================================================
# 6. CORRELATION MATRIX
# ============================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = (
    cleaned_df[
        correlation_columns
    ].corr()
)


plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Titanic Numeric Correlation Matrix"
)

save_plot(
    "05_correlation_heatmap.png"
)


# ============================================================
# FIND TWO STRONGEST CORRELATIONS
# ============================================================

pairs = []

for i in range(
    len(correlation_columns)
):

    for j in range(
        i + 1,
        len(correlation_columns)
    ):

        column_1 = correlation_columns[i]
        column_2 = correlation_columns[j]

        value = correlation_matrix.loc[
            column_1,
            column_2
        ]

        pairs.append(
            (
                column_1,
                column_2,
                value,
                abs(value)
            )
        )


pairs = sorted(
    pairs,
    key=lambda x: x[3],
    reverse=True
)

strongest_1 = pairs[0]
strongest_2 = pairs[1]


# ============================================================
# 7. MULTIVARIATE DATA STORY
# ============================================================

# Chart 1: survival by sex and class

plt.figure(
    figsize=(9, 6)
)

sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Passenger Class and Sex"
)

plt.ylabel(
    "Survival Rate"
)

save_plot(
    "06_survival_by_sex_class.png"
)


# Chart 2: age distribution by survival

plt.figure(
    figsize=(9, 6)
)

sns.boxplot(
    data=cleaned_df,
    x="survived",
    y="age"
)

plt.title(
    "Age Distribution by Survival"
)

plt.xlabel(
    "Survived"
)

save_plot(
    "07_age_by_survival.png"
)


# Chart 3: fare by passenger class and survival

plt.figure(
    figsize=(9, 6)
)

sns.boxplot(
    data=cleaned_df,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title(
    "Fare by Passenger Class and Survival"
)

save_plot(
    "08_fare_class_survival.png"
)


# Chart 4: family size and survival

story_df = cleaned_df.copy()

story_df["family_size"] = (
    story_df["sibsp"]
    + story_df["parch"]
    + 1
)

plt.figure(
    figsize=(10, 6)
)

sns.barplot(
    data=story_df,
    x="family_size",
    y="survived"
)

plt.title(
    "Survival Rate by Family Size"
)

plt.xlabel(
    "Family Size"
)

plt.ylabel(
    "Survival Rate"
)

save_plot(
    "09_family_size_survival.png"
)


# ============================================================
# 8. STANDARDIZATION CHECK
# ============================================================

standardizer = StandardScaler()

standardized_values = standardizer.fit_transform(
    cleaned_df[
        ["age", "fare"]
    ]
)

standardized_df = pd.DataFrame(
    standardized_values,
    columns=[
        "age_standardized",
        "fare_standardized"
    ]
)


standardization_summary = pd.DataFrame(
    {
        "original_mean": [
            cleaned_df["age"].mean(),
            cleaned_df["fare"].mean()
        ],
        "original_std": [
            cleaned_df["age"].std(),
            cleaned_df["fare"].std()
        ],
        "standardized_mean": [
            standardized_df[
                "age_standardized"
            ].mean(),
            standardized_df[
                "fare_standardized"
            ].mean()
        ],
        "standardized_std": [
            standardized_df[
                "age_standardized"
            ].std(),
            standardized_df[
                "fare_standardized"
            ].std()
        ]
    },
    index=[
        "age",
        "fare"
    ]
)


print("\nSTANDARDIZATION CHECK")
print("-" * 70)

print(
    standardization_summary
)


# ============================================================
# STANDARDIZATION VISUAL CHECK
# ============================================================

plt.figure(
    figsize=(9, 5)
)

sns.kdeplot(
    cleaned_df["age"],
    label="Age - original"
)

sns.kdeplot(
    standardized_df["age_standardized"],
    label="Age - standardized"
)

plt.title(
    "Age Before and After Standardization"
)

plt.legend()

save_plot(
    "10_age_standardization.png"
)


plt.figure(
    figsize=(9, 5)
)

sns.kdeplot(
    cleaned_df["fare"],
    label="Fare - original"
)

sns.kdeplot(
    standardized_df["fare_standardized"],
    label="Fare - standardized"
)

plt.title(
    "Fare Before and After Standardization"
)

plt.legend()

save_plot(
    "11_fare_standardization.png"
)


# ============================================================
# 9. WRITE EDA REPORT
# ============================================================

report = []

report.append(
    "# Titanic EDA Report"
)

report.append(
    "\n## Dataset Profile"
)

report.append(
    f"- Dataset shape after cleaning: "
    f"`{cleaned_df.shape}`."
)

report.append(
    "\n### Missing-value strategy"
)

report.extend(
    cleaning_notes
)

report.append(
    "\n## Univariate Analysis"
)

report.append(
    f"- Age IQR outlier count: "
    f"**{age_outliers}**."
)

report.append(
    f"- Fare IQR outlier count: "
    f"**{fare_outliers}**."
)

report.append(
    f"- Fare mean: **{fare_mean:.3f}**."
)

report.append(
    f"- Fare median: **{fare_median:.3f}**."
)

report.append(
    f"- Fare mode: **{fare_mode:.3f}**."
)

report.append(
    f"- **Interpretation:** {fare_skew_text}"
)

report.append(
    "\n## Bivariate Analysis"
)

report.append(
    f"- Female survival rate: "
    f"**{female_survival:.3f}**."
)

report.append(
    f"- Male survival rate: "
    f"**{male_survival:.3f}**."
)

report.append(
    f"- Female + first-class survival rate: "
    f"**{female_first_class:.3f}**."
)

report.append(
    f"- Male + third-class survival rate: "
    f"**{male_third_class:.3f}**."
)

report.append(
    "\n### Correlation"
)

report.append(
    f"- Strongest correlation pair: "
    f"`{strongest_1[0]}` and `{strongest_1[1]}` "
    f"with correlation **{strongest_1[2]:.3f}**."
)

report.append(
    f"- Second strongest correlation pair: "
    f"`{strongest_2[0]}` and `{strongest_2[1]}` "
    f"with correlation **{strongest_2[2]:.3f}**."
)

report.append(
    "\n## Multivariate Data Story"
)

report.append(
    "\n### Chart 1 — Survival by Sex and Passenger Class"
)

report.append(
    "The chart compares survival rates across passenger classes "
    "separately for males and females. It allows the interaction "
    "between sex and passenger class to be examined rather than "
    "looking at either variable independently."
)

report.append(
    "\n### Chart 2 — Age and Survival"
)

report.append(
    "The box plot compares the age distributions of passengers "
    "who survived and those who did not. Differences in the "
    "central tendency and spread provide an additional view "
    "of the relationship between age and survival."
)

report.append(
    "\n### Chart 3 — Fare, Class and Survival"
)

report.append(
    "Fare is examined together with passenger class and survival. "
    "Because fare and passenger class are related, this visualization "
    "helps distinguish the effect of fare levels within the class structure."
)

report.append(
    "\n### Chart 4 — Family Size and Survival"
)

report.append(
    "Family size is calculated from siblings/spouses plus parents/children "
    "plus the passenger. The chart shows how survival rates vary across "
    "different family-size groups."
)

report.append(
    "\n## Standardization Check"
)

report.append(
    "Age and fare were standardized using the z-score transformation. "
    "The resulting means are approximately zero and the standard deviations "
    "are approximately one, confirming that the exploratory standardization "
    "was applied correctly. This transformation is only an EDA sanity check "
    "and is not used as the modeling pipeline's preprocessing."
)

report.append(
    "\n## Output Files"
)

report.append(
    "- `titanic.csv` — cleaned offline dataset."
)

report.append(
    "- `outputs/` — supporting EDA charts."
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report)
    )


print(
    f"\nEDA report saved to: {REPORT_FILE}"
)

print(
    "\n01_eda.py completed."
)
