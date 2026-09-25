import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(__file__)

DATA_FILE = os.path.join(
    BASE_DIR,
    "titanic.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD THE OFFLINE DATASET
# ============================================================

print("=" * 80)
print("LOADING TITANIC DATASET")
print("=" * 80)

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        "analytics/titanic.csv was not found. "
        "Run 01_eda.py first so that the offline dataset is created."
    )

df = pd.read_csv(
    DATA_FILE
)

print(
    f"Dataset shape: {df.shape}"
)

print(
    "\nColumns:"
)

print(
    list(df.columns)
)


# ============================================================
# 3. CLASS BALANCE
# ============================================================

print("\n" + "=" * 80)
print("CLASS BALANCE")
print("=" * 80)

class_counts = (
    df["survived"]
    .value_counts()
    .sort_index()
)

class_percentages = (
    df["survived"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

class_balance = pd.DataFrame(
    {
        "count": class_counts,
        "percentage": class_percentages
    }
)

print(
    class_balance
)

class_balance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "class_balance.csv"
    )
)


# ============================================================
# 4. CLASSIFICATION DATA
# ============================================================

TARGET = "survived"

# Use useful passenger/customer-style features.
# Fare is excluded because it is the target of the regression task
# and is not necessary for this classification model.

classification_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

X = df[
    classification_features
].copy()

y = df[
    TARGET
].copy()


# ============================================================
# 5. STRATIFIED TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("STRATIFIED TRAIN / TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)

print(
    "\nStratification is used because the target contains "
    "survived/not-survived classes. It keeps approximately "
    "the same class proportions in both train and test sets."
)


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

categorical_features = [
    "sex",
    "embarked"
]


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 7. THREE CLASSIFIERS
# ============================================================

logistic_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


decision_tree_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        )
    ]
)


random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                oob_score=True
            )
        )
    ]
)


models = {
    "Logistic Regression": logistic_pipeline,
    "Decision Tree": decision_tree_pipeline,
    "Random Forest": random_forest_pipeline
}


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate_classifier(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    cm = confusion_matrix(
        y_test,
        predictions
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC AUC  : {auc:.4f}"
    )

    # Save confusion matrix
    plt.figure(
        figsize=(5, 4)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Survived",
            "Survived"
        ]
    ).plot()

    plt.title(
        f"{name} - Confusion Matrix"
    )

    plt.tight_layout()

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_confusion_matrix.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"{name} - ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_roc.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc,
        "Pipeline": model
    }


# ============================================================
# 9. TRAIN AND EVALUATE ALL THREE MODELS
# ============================================================

results = []

for model_name, model in models.items():

    result = evaluate_classifier(
        model_name,
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(
        result
    )


classification_results = pd.DataFrame(
    [
        {
            key: value
            for key, value in result.items()
            if key != "Pipeline"
        }
        for result in results
    ]
)

print("\n" + "=" * 80)
print("CLASSIFICATION COMPARISON")
print("=" * 80)

print(
    classification_results
)

classification_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_results.csv"
    ),
    index=False
)


# ============================================================
# 10. DECISION TREE VISUALIZATION
# ============================================================

print("\n" + "=" * 80)
print("DECISION TREE VISUALIZATION")
print("=" * 80)

tree_model = decision_tree_pipeline

tree_preprocessor = tree_model.named_steps[
    "preprocessor"
]

tree_classifier = tree_model.named_steps[
    "classifier"
]

feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)

plt.figure(
    figsize=(24, 12)
)

plot_tree(
    tree_classifier,
    feature_names=feature_names,
    class_names=[
        "Not Survived",
        "Survived"
    ],
    filled=True,
    rounded=True,
    max_depth=4,
    fontsize=8
)

plt.title(
    "Decision Tree Classifier"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "decision_tree.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. IMBALANCE COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("IMBALANCE HANDLING COMPARISON")
print("=" * 80)


# ------------------------------------------------------------
# A. BASELINE
# ------------------------------------------------------------

baseline_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ------------------------------------------------------------
# B. CLASS WEIGHT BALANCED
# ------------------------------------------------------------

balanced_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ------------------------------------------------------------
# C. SMOTE
# ------------------------------------------------------------

smote_pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            __import__(
                "imblearn.over_sampling",
                fromlist=["SMOTE"]
            ).SMOTE(
                random_state=42
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


imbalance_models = {
    "Baseline": baseline_pipeline,
    "Class Weight Balanced": balanced_pipeline,
    "SMOTE": smote_pipeline
}

imbalance_results = []

for name, model in imbalance_models.items():

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    imbalance_results.append(
        {
            "Strategy": name,
            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "F1": f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        }
    )


imbalance_df = pd.DataFrame(
    imbalance_results
)

print(
    imbalance_df
)

imbalance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "imbalance_comparison.csv"
    ),
    index=False
)


# ============================================================
# 12. RANDOM FOREST GRID SEARCH + OOB
# ============================================================

print("\n" + "=" * 80)
print("RANDOM FOREST GRID SEARCH")
print("=" * 80)


rf_for_search = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


param_grid = {
    "classifier__n_estimators": [
        100,
        200
    ],
    "classifier__max_depth": [
        None,
        5,
        10
    ],
    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}


cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    estimator=rf_for_search,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring="f1",
    n_jobs=-1,
    return_train_score=False
)


grid_search.fit(
    X_train,
    y_train
)


best_rf_pipeline = grid_search.best_estimator_

best_rf_classifier = (
    best_rf_pipeline
    .named_steps["classifier"]
)


print(
    "\nBest parameters:"
)

print(
    grid_search.best_params_
)

print(
    f"\nBest cross-validation F1: "
    f"{grid_search.best_score_:.4f}"
)

print(
    f"OOB score: "
    f"{best_rf_classifier.oob_score_:.4f}"
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "grid_search_results.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Random Forest GridSearchCV Results\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Best Parameters:\n"
    )

    file.write(
        str(
            grid_search.best_params_
        )
    )

    file.write(
        "\n\n"
    )

    file.write(
        f"Best CV F1: "
        f"{grid_search.best_score_:.4f}\n"
    )

    file.write(
        f"OOB Score: "
        f"{best_rf_classifier.oob_score_:.4f}\n"
    )


# ============================================================
# 13. REGRESSION TASK — PREDICT FARE
# ============================================================

print("\n" + "=" * 80)
print("REGRESSION TASK")
print("=" * 80)


regression_target = "fare"

regression_features = [
    column
    for column in df.columns
    if column != regression_target
    and column not in [
        "adult_male",
        "alone"
    ]
]


X_reg = df[
    regression_features
].copy()

y_reg = df[
    regression_target
].copy()


# Separate numeric and categorical columns automatically.

reg_numeric_features = (
    X_reg
    .select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)

reg_categorical_features = (
    X_reg
    .select_dtypes(
        exclude=np.number
    )
    .columns
    .tolist()
)


reg_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


reg_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            reg_numeric_pipeline,
            reg_numeric_features
        ),
        (
            "categorical",
            reg_categorical_pipeline,
            reg_categorical_features
        )
    ]
)


X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)


regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


reg_predictions = regression_pipeline.predict(
    X_reg_test
)


mae = mean_absolute_error(
    y_reg_test,
    reg_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        reg_predictions
    )
)

r2 = r2_score(
    y_reg_test,
    reg_predictions
)


n = len(
    y_reg_test
)

p = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(X_reg_test)
    .shape[1]
)


if n - p - 1 > 0:

    adjusted_r2 = (
        1
        -
        (
            (1 - r2)
            *
            (n - 1)
            /
            (n - p - 1)
        )
    )

else:

    adjusted_r2 = np.nan


print(
    f"MAE         : {mae:.4f}"
)

print(
    f"RMSE        : {rmse:.4f}"
)

print(
    f"R²          : {r2:.4f}"
)

print(
    f"Adjusted R² : {adjusted_r2:.4f}"
)


# ============================================================
# 14. RESIDUAL PLOT + HETEROSCEDASTICITY CHECK
# ============================================================

residuals = (
    y_reg_test.to_numpy()
    -
    reg_predictions
)

absolute_residuals = np.abs(
    residuals
)

fitted_values = (
    reg_predictions
)

residual_fitted_correlation = np.corrcoef(
    fitted_values,
    absolute_residuals
)[0, 1]


plt.figure(
    figsize=(8, 6)
)

sns.scatterplot(
    x=fitted_values,
    y=residuals
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Fitted Values"
)

plt.ylabel(
    "Residuals"
)

plt.title(
    "Regression Residual Plot"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "regression_residual_plot.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()


if (
    abs(residual_fitted_correlation)
    >= 0.20
):

    heteroscedasticity_conclusion = (
        "The residual plot shows suggestive evidence "
        "of heteroscedasticity because the absolute "
        "residuals have a noticeable relationship with "
        "the fitted values. This should be treated as "
        "diagnostic evidence rather than a formal test."
    )

else:

    heteroscedasticity_conclusion = (
        "The residual plot does not show strong evidence "
        "of heteroscedasticity based on the relationship "
        "between fitted values and absolute residuals. "
        "The residual spread appears reasonably stable, "
        "although this visual diagnostic is not a formal test."
    )


print(
    "\nHeteroscedasticity conclusion:"
)

print(
    heteroscedasticity_conclusion
)


# ============================================================
# 15. FINAL MODEL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)


final_comparison = classification_results[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC"
    ]
].copy()


final_comparison["Regression_MAE"] = np.nan
final_comparison["Regression_RMSE"] = np.nan
final_comparison["Regression_R2"] = np.nan
final_comparison["Regression_Adjusted_R2"] = np.nan


regression_row = pd.DataFrame(
    [
        {
            "Model": "Linear Regression",
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "AUC": np.nan,
            "Regression_MAE": mae,
            "Regression_RMSE": rmse,
            "Regression_R2": r2,
            "Regression_Adjusted_R2": adjusted_r2
        }
    ]
)


final_comparison = pd.concat(
    [
        final_comparison,
        regression_row
    ],
    ignore_index=True
)


print(
    final_comparison
)


final_comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 16. SELECT FINAL CLASSIFICATION PIPELINE
# ============================================================

# Use F1 as the primary selection metric because the assignment
# explicitly requires precision, recall and F1 for the classification
# task. The selected object is a complete preprocessing + estimator
# pipeline.

best_model_row = (
    classification_results
    .sort_values(
        "F1",
        ascending=False
    )
    .iloc[0]
)

best_model_name = (
    best_model_row["Model"]
)

best_model = next(
    result["Pipeline"]
    for result in results
    if result["Model"] == best_model_name
)


# Refit the selected complete pipeline on the training data.
best_model.fit(
    X_train,
    y_train
)


# ============================================================
# 17. SAVE COMPLETE PIPELINE
# ============================================================

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_classifier_pipeline.joblib"
)

joblib.dump(
    best_model,
    MODEL_FILE
)

print(
    f"\nComplete pipeline saved to:"
)

print(
    MODEL_FILE
)


# ============================================================
# 18. RELOAD AND TEST RAW INPUT
# ============================================================

print("\n" + "=" * 80)
print("RELOADING SAVED PIPELINE")
print("=" * 80)

loaded_pipeline = joblib.load(
    MODEL_FILE
)


raw_sample = pd.DataFrame(
    [
        {
            "pclass": 3,
            "sex": "male",
            "age": 30,
            "sibsp": 0,
            "parch": 0,
            "embarked": "S"
        }
    ]
)


loaded_prediction = (
    loaded_pipeline
    .predict(raw_sample)
)


print(
    "\nRaw sample:"
)

print(
    raw_sample
)

print(
    "\nReloaded pipeline prediction:"
)

print(
    loaded_prediction
)


# ============================================================
# 19. WRITE MODELING REPORT
# ============================================================

report_lines = []

report_lines.append(
    "# Predictive Modeling Report"
)

report_lines.append(
    "\n## Train/Test Split"
)

report_lines.append(
    "A stratified 80/20 train/test split was used for "
    "classification. Stratification preserves approximately "
    "the observed survived/not-survived class proportions "
    "in both partitions."
)

report_lines.append(
    "\n## Preprocessing"
)

report_lines.append(
    "Numeric features use median imputation followed by "
    "StandardScaler. Categorical features use most-frequent "
    "imputation followed by one-hot encoding. The "
    "ColumnTransformer is inside each modeling Pipeline, "
    "so preprocessing is fitted only on the training data "
    "and then applied to the test data."
)

report_lines.append(
    "\n## Classification Models"
)

for _, row in classification_results.iterrows():

    report_lines.append(
        f"\n### {row['Model']}"
    )

    report_lines.append(
        f"- Accuracy: **{row['Accuracy']:.4f}**"
    )

    report_lines.append(
        f"- Precision: **{row['Precision']:.4f}**"
    )

    report_lines.append(
        f"- Recall: **{row['Recall']:.4f}**"
    )

    report_lines.append(
        f"- F1: **{row['F1']:.4f}**"
    )

    report_lines.append(
        f"- AUC: **{row['AUC']:.4f}**"
    )


report_lines.append(
    "\n## Imbalance Handling"
)

report_lines.append(
    imbalance_df.to_string(
        index=False
    )
)

best_imbalance = (
    imbalance_df
    .sort_values(
        "F1",
        ascending=False
    )
    .iloc[0]
)

report_lines.append(
    f"\nBased on F1 on the held-out test set, "
    f"the highest observed value among the three "
    f"imbalance strategies was produced by "
    f"**{best_imbalance['Strategy']}** "
    f"({best_imbalance['F1']:.4f}). "
    f"The comparison should also consider the precision/recall "
    f"trade-off rather than F1 alone."
)


report_lines.append(
    "\n## Random Forest Hyperparameter Tuning"
)

report_lines.append(
    f"- Best parameters: "
    f"`{grid_search.best_params_}`"
)

report_lines.append(
    f"- Best cross-validation F1: "
    f"**{grid_search.best_score_:.4f}**"
)

report_lines.append(
    f"- OOB score: "
    f"**{best_rf_classifier.oob_score_:.4f}**"
)


report_lines.append(
    "\n## Regression"
)

report_lines.append(
    f"- MAE: **{mae:.4f}**"
)

report_lines.append(
    f"- RMSE: **{rmse:.4f}**"
)

report_lines.append(
    f"- R²: **{r2:.4f}**"
)

report_lines.append(
    f"- Adjusted R²: **{adjusted_r2:.4f}**"
)

report_lines.append(
    f"- Heteroscedasticity assessment: "
    f"{heteroscedasticity_conclusion}"
)


report_lines.append(
    "\n## Final Classifier Selection"
)

report_lines.append(
    f"The classifier selected for the saved deployment "
    f"artifact was **{best_model_name}**, based on the "
    f"highest observed held-out F1 score among the three "
    f"classifiers. Its metrics were Accuracy "
    f"**{best_model_row['Accuracy']:.4f}**, Precision "
    f"**{best_model_row['Precision']:.4f}**, Recall "
    f"**{best_model_row['Recall']:.4f}**, F1 "
    f"**{best_model_row['F1']:.4f}**, and AUC "
    f"**{best_model_row['AUC']:.4f}**."
)

report_lines.append(
    "\nThe saved artifact contains the complete preprocessing "
    "steps together with the classifier, rather than the "
    "bare estimator. It was reloaded with joblib and tested "
    "using a raw, unprocessed passenger record."
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "modeling_report.md"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report_lines)
    )


print(
    "\nModeling report saved."
)

print(
    "\n02_modeling.py completed successfully."
)
