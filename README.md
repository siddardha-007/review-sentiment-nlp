#  Flipkart Product Review Sentiment Analyzer

An end-to-end Natural Language Processing (NLP) and Machine Learning web application that classifies customer product reviews into **Positive**, **Neutral**, or **Negative** sentiment using **TF-IDF** and a calibrated **Linear Support Vector Machine (`LinearSVC`)**.

---

##  Features
- **Automated Text Preprocessing:** Handles multi-encoding CSV files, removes duplicates to prevent data leakage, preserves negation words (`not`, `don't`), and applies WordNet lemmatisation.
- **Model Benchmarking & Tuning:** Automatically compares **Multinomial Naive Bayes**, **Logistic Regression**, and **LinearSVC** using Macro F1-Score, then tunes the winning pipeline via `GridSearchCV`.
- **Interactive Web Interface:** Built with **Streamlit**, displaying real-time sentiment predictions, class confidence breakdowns, and the key vocabulary terms detected in each review.

---

##  Tech Stack
- **Language:** Python 3.10+
- **Data Processing & NLP:** Pandas, NumPy, NLTK, Regex
- **Machine Learning:** Scikit-learn (`TfidfVectorizer`, `LinearSVC`, `CalibratedClassifierCV`, `GridSearchCV`)
- **Web UI:** Streamlit

---

##  Project Structure
```text
flipkart-ml-sentiment/
├── data/
│   └── flipkart_reviews.csv      # Dataset (from Kaggle)
├── models/
│   └── best_sentiment_model.pkl  # Saved trained ML pipeline
├── src/
│   ├── preprocess.py             # Data cleaning, deduplication & lemmatisation
│   └── train.py                  # Model comparison, hyperparameter tuning & saving
├── app.py                        # Streamlit web application
├── requirements.txt              # Project dependencies
└── README.md
```

##  Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/siddardha-007/review-sentiment-nlp.git
cd review-sentiment-nlp
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Add the Dataset
Download the **Flipkart Product Reviews** dataset from Kaggle and save it to the following path:
```text
data/flipkart_reviews.csv
```

### 4. Train the Model
Run the training script to preprocess the data, benchmark the ML models, tune hyperparameters, and save the winning pipeline:
```bash
python src/train.py
```

### 5. Launch the Web App
Start the interactive Streamlit interface:
```bash
streamlit run app.py
```

---

## 📊 Methodology & Best Practices

* **Label Mapping:** Converts star ratings into three distinct sentiment classes:
  * **1–2 Stars** $\rightarrow$ Negative
  * **3 Stars** $\rightarrow$ Neutral
  * **4–5 Stars** $\rightarrow$ Positive
* **Feature Engineering:** Uses TF-IDF with unigrams and bigrams (`ngram_range=(1, 2)`) and sublinear term frequency scaling so multi-word expressions like `"not good"` or `"value money"` are captured accurately.
* **Class Imbalance Handling:** Applies `class_weight="balanced"` during training and evaluates models using **Macro F1-Score** rather than raw accuracy so minority classes (Neutral and Negative) are predicted reliably.