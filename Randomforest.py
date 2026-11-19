from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main() -> None:
    dataset_path = Path(__file__).with_name("people-100.csv")
    df = pd.read_csv(dataset_path)

    required_columns = {"Sex", "Date of birth", "Job Title"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    df = df.dropna(subset=["Sex"]).copy()
    if df.empty:
        raise ValueError("The dataset has no rows with a target value in 'Sex'.")

    date_of_birth = pd.to_datetime(df["Date of birth"], errors="coerce")
    df["Age"] = (pd.Timestamp.today().normalize() - date_of_birth).dt.days / 365.2425


    # Keep useful, non-identifying features; exclude names, contact details, and IDs.
    features = df[["Age", "Job Title"]]
    target = df["Sex"]

    numeric_features = ["Age"]
    categorical_features = ["Job Title"]
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(n_estimators=200, random_state=42),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Dataset: {dataset_path.name}")
    print(f"Rows used: {len(df)}")
    print(f"Training rows: {len(X_train)}; test rows: {len(X_test)}")
    print(f"Target classes: {', '.join(map(str, model.classes_))}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(
        pd.DataFrame(
            confusion_matrix(y_test, predictions, labels=model.classes_),
            index=model.classes_,
            columns=model.classes_,
        )
    )


if __name__ == "__main__":
    main()
