from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Continuous numerical features.
NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

# Discrete/categorical features.
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def create_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Create the preprocessing pipeline.

    Numerical features:
        - Median imputation
        - Optional standard scaling

    Categorical/discrete features:
        - Most-frequent imputation
        - One-hot encoding

    Parameters
    ----------
    scale_numeric:
        Whether numerical features should be standardized.

    Returns
    -------
    ColumnTransformer
        Leakage-safe preprocessing transformer.
    """

    if scale_numeric:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def create_logistic_regression_pipeline() -> Pipeline:
    """
    Create the complete Logistic Regression pipeline.
    """

    preprocessor = create_preprocessor(scale_numeric=True)

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def create_random_forest_pipeline() -> Pipeline:
    """
    Create the complete Random Forest pipeline.
    """

    preprocessor = create_preprocessor(scale_numeric=False)

    classifier = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline