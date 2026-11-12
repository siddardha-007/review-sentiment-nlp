import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score
from preprocess import load_flipkart_data

DATA_PATH = "data/flipkart_reviews.csv"
MODEL_SAVE_PATH = "models/best_sentiment_model.pkl"

def train_and_evaluate():
    print("Loading and cleaning Flipkart dataset...")
    df = load_flipkart_data(DATA_PATH)
    print(f"Dataset ready: {len(df)} unique reviews.")
    print(df["sentiment"].value_counts(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"]
    )

    # 1. Compare 3 top classical NLP algorithms
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Linear SVC (SVM)": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=42)
        )
    }

    best_score = 0
    best_model_name = ""

    print("--- Benchmarking Classical ML Models ---")
    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)),
            ("clf", clf)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        score = f1_score(y_test, preds, average="macro")
        print(f"{name:25s} | Macro F1-Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model_name = name

    print(f"\nWinner: {best_model_name}. Tuning hyperparameters...")

    # 2. Tune the winning LinearSVC pipeline with GridSearchCV
    final_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42)))
    ])

    param_grid = {
        "tfidf__max_features": [10000, 20000],
        "tfidf__min_df": [2, 5],
        "clf__estimator__C": [0.5, 1.0]
    }

    grid = GridSearchCV(final_pipeline, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    final_preds = best_pipeline.predict(X_test)

    print(f"\nBest Hyperparameters: {grid.best_params_}")
    print("\n--- Final Classification Report on Test Set ---")
    print(classification_report(y_test, final_preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)
    print(f"Model saved successfully to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_and_evaluate()