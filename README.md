# Campus Desk — NLP University Student Chatbot (End-to-End)

An intent-classification chatbot trained on 10,000 university student Q&A pairs
across 30 intents (fees, hostel, admissions, exams, placements, library, etc.),
served through a Flask web app.

## Project structure
```
chatbot_project/
├── data/
│   └── chatbot_dataset.csv          # source dataset
├── preprocess.py                    # text cleaning utilities
├── train.py                         # full training pipeline (script version)
├── build_notebook.py                # generates the executed .ipynb
├── University_Student_Chatbot_EndToEnd.ipynb   # ★ the notebook you asked for (already run)
├── models/                          # saved artifacts (created by train.py)
│   ├── tfidf_vectorizer.pkl
│   ├── intent_classifier.pkl
│   ├── label_encoder.pkl
│   ├── response_map.json
│   └── metadata.json
├── outputs/                         # evaluation charts/reports
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── model_comparison.csv
│   └── classification_report.csv
├── app/
│   ├── app.py                       # Flask server
│   └── templates/index.html         # chat UI
└── requirements.txt
```

## 1. Setup
```bash
cd chatbot_project
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

## 2. Train the model (regenerates everything in `models/` and `outputs/`)
```bash
python train.py
```
Or open **`University_Student_Chatbot_EndToEnd.ipynb`** in Jupyter — it contains
the exact same pipeline, cell-by-cell, already executed with outputs (EDA charts,
model comparison, confusion matrix, live chatbot test).

## 3. Run the chatbot web app
```bash
python app/app.py
```
Open **http://127.0.0.1:5000** in your browser and start chatting.

## How it works
1. **Preprocessing** — lowercase, strip punctuation/digits/URLs, remove template
   filler words ("please", "I need help", etc.), which the synthetic dataset
   injects around every question.
2. **Feature extraction** — TF-IDF with uni+bi-grams (~1,000-word vocabulary).
3. **Model** — Logistic Regression chosen after benchmarking against Linear SVM,
   Multinomial Naive Bayes, and Random Forest (all scored via 5-fold CV + held-out test).
4. **Response generation** — retrieval-based: predicted intent maps to the
   canonical response for that intent (`response_map.json`).
5. **Confidence gating** — if the model's top prediction probability is below
   `0.35`, the bot asks the user to rephrase instead of guessing (handles
   out-of-scope questions gracefully).

## Model performance
See `outputs/classification_report.csv` and `outputs/confusion_matrix.png`.
Test accuracy is ~99–100% because the dataset is synthetically templated with
distinctive keywords per intent — expect 85–95% on real, messy student phrasing.
Retrain periodically on logged real conversations for production use.

## Next steps for production
- Log real queries → active-learning retraining loop.
- Swap TF-IDF+LogReg for `sentence-transformers` embeddings if you need semantic
  (not just keyword) matching.
- Add multi-turn context/slot-filling for follow-ups ("and for PG students?").
- Deploy behind Gunicorn/Nginx (dev server used here is not production-grade).
- Add authentication + rate limiting before exposing publicly.
