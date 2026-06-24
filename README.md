# Fake News Detection — NLP Classification

A text classification project that detects fake vs. real news articles by comparing a classical machine learning approach (TF-IDF + Logistic Regression) against a fine-tuned transformer model (BERT).

Built as part of the **Intern Alpha Data Science Internship**.

## Problem Statement

Misinformation spreads fast, and manually verifying news is not scalable. This project builds and compares two NLP approaches to automatically classify a news article as **real** or **fake** based on its title and text content.

## Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle, by Clément Bisaillon)

- 8,735 articles total after cleaning (4,621 fake / 4,114 real)
- Columns: `title`, `text`, `subject`, `date`
- Source skew: all fake articles are labeled `subject = News`, all real articles are labeled `subject = politicsNews`. This makes `subject` a direct label leak, so it was **deliberately excluded** from modeling — using it would have produced a trivial, meaningless 100% accuracy.

## Approach

### 1. Exploratory Data Analysis
- Checked class balance (53% fake / 47% real — close enough to use accuracy as a meaningful metric)
- Compared word count distributions: fake articles average ~449 words vs. ~396 for real articles
- Compared top words by class:
  - **Real news** is dominated by formal, attribution-style language (`said`, `reuters`, `washington`, `senate`, `administration`) — consistent with wire-service journalism
  - **Fake news** leans on social/blog-style terms (`twitter`, `image`, `getty`, `featured`) — several of which turned out to be scraping artifacts (image captions/credits) rather than genuine signal, and were removed during cleaning

### 2. Preprocessing
- Lowercased text, removed URLs/punctuation/numbers
- Removed standard English stopwords **plus** dataset-specific noise words identified during EDA (`reuters`, `getty`, `image`, `images`, `featured`, `realdonaldtrump`) to avoid the model learning scraping artifacts instead of language patterns
- Combined `title` + `text` into a single `content` field

### 3. Modeling
Two models were trained and compared:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression (TF-IDF, 5,000 features) | 0.9874 | 0.9878 | 0.9854 | 0.9866 |
| BERT (fine-tuned, `bert-base-uncased`) | 0.9900 | 0.9658 | 1.0000 | 0.9826 |

*Note: Logistic Regression was evaluated on the full test set (1,747 articles). BERT was fine-tuned and evaluated on a stratified 4,000-article subsample (800 test articles) due to compute/time constraints on free-tier GPU. The two results are not on identical test sets, and this is stated explicitly for transparency.*

### 4. Key Findings
- A simple linear model (Logistic Regression on TF-IDF) already performs extremely well (~98.7%), indicating that in this dataset, fake vs. real news is strongly correlated with **writing style and source conventions** rather than purely semantic "truthfulness" — a known property of this dataset, and an important caveat rather than a hidden one.
- BERT achieved a marginally higher accuracy (99%) with a notably different error profile: **100% recall on real news** (zero real articles misclassified as fake) at a small cost to precision (a few more fake articles misclassified as real).
- This tradeoff is practically meaningful: depending on the deployment context (e.g., suppressing real journalism vs. letting some fake articles through), one model's error profile may be preferable to the other even at similar overall accuracy.

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (TF-IDF, Logistic Regression, metrics)
- PyTorch + Hugging Face Transformers (BERT fine-tuning)
- Matplotlib, Seaborn (visualization)
- NLTK (stopword removal)

## Repository Structure
```
fake-news-detection/
├── data/                       # Fake.csv, True.csv (raw data)
├── notebooks/
│   └── fake_news_detection.ipynb
├── images/
│   ├── top_words_comparison.png
│   ├── confusion_matrix_lr.png
│   └── model_comparison.png
└── README.md
```

## How to Run
1. Clone this repo and open `notebooks/fake_news_detection.ipynb` in Google Colab or Jupyter
2. Place `Fake.csv` and `True.csv` (from the Kaggle link above) in the `data/` folder
3. Run all cells in order (GPU recommended for the BERT section)

## Future Improvements
- Train BERT on the full dataset rather than a subsample, given more compute time
- Try other transformer variants (RoBERTa, DistilBERT) for a speed/accuracy comparison
- Add SHAP/LIME-based interpretability to explain individual predictions
- Deploy as a live demo via Streamlit or Hugging Face Spaces

## Author
Divya Kauluri — Data Science Intern, Intern Alpha
