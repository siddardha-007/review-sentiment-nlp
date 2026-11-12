import re
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer

# Download WordNet dictionary once for lemmatisation
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Cleans review text while preserving negation words like 'not' and 'don't'."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|<.*?>", "", text)
    # Keep letters and apostrophes so "don't" stays intact
    text = re.sub(r"[^a-z\s']", " ", text)
    
    # Lemmatise words (reduces vocabulary noise without losing meaning)
    words = [lemmatizer.lemmatize(w) for w in text.split() if len(w) > 1]
    return " ".join(words)

def load_flipkart_data(filepath: str, max_rows: int = 25000) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="latin1", encoding_errors="replace")
    df.columns = [c.strip().lower() for c in df.columns]

    # Combine review summary/title and detailed review text
    title_col = "title" if "title" in df.columns else df.columns[0]
    review_col = "review" if "review" in df.columns else df.columns[1]
    rating_col = "rating" if "rating" in df.columns else "rate"

    df["full_text"] = (df[title_col].fillna("") + " " + df[review_col].fillna("")).str.strip()
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    df = df.dropna(subset=[rating_col])

    # Remove empty rows and exact duplicate reviews to prevent data leakage
    df = df[df["full_text"].str.len() > 3].drop_duplicates(subset=["full_text"])

    # Map 1-5 ratings into 3 classes: Negative (0), Neutral (1), Positive (2)
    def label_mapper(r):
        if r <= 2:
            return "Negative"
        elif r == 3:
            return "Neutral"
        return "Positive"

    df["sentiment"] = df[rating_col].apply(label_mapper)
    df["cleaned_text"] = df["full_text"].apply(clean_text)


    return df.reset_index(drop=True)