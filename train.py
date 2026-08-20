"""
train.py
--------
End-to-end training pipeline for the University Student Chatbot (Intent Classification).

Pipeline:
1. Load & clean data
2. Train/test split (stratified)
3. TF-IDF vectorization
4. Train & compare multiple models
5. Pick best model, evaluate in depth
6. Build intent -> response lookup (retrieval-based answer generation)
7. Save all artifacts (vectorizer, model, label encoder, response map) for deployment
"""

import json
import time
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

from preprocess import clean_text

DATA_PATH = "data/chatbot_dataset.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df = df.dropna(subset=["question", "response", "intent"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows | {df['intent'].nunique()} intents | "
          f"{df['category'].nunique()} categories")
    return df


ENHANCED_RESPONSES = {
    "courses": "Undergraduate (UG) Programs: B.Tech (CSE, ECE, Mechanical, Civil - 4 Years), BBA (General, Finance, Marketing - 3 Years), B.Sc (Computer Science, Physics, Chemistry, Math - 3 Years), B.Com, and BA.\n\nPostgraduate (PG) Programs: M.Sc (Data Science, Computer Science, Chemistry, Physics - 2 Years), MCA (2 Years), MBA (2 Years), and M.Tech (2 Years). All programs follow CBCS semester curriculum.",
    "eligibility": "Undergraduate (UG) Eligibility: 10+2 / Higher Secondary with minimum 50-60% aggregate (PCM for B.Tech, relevant science subjects for B.Sc, any stream for BBA / B.Com).\n\nPostgraduate (PG) Eligibility: Bachelor's degree in relevant discipline with minimum 50-55% marks (e.g., B.Sc/BCA for M.Sc/MCA, Any Graduate for MBA). Final-year students can apply provisionally.",
    "admission": "To apply for new admission (UG / PG): 1. Visit the official Admission Portal. 2. Register with your email/phone. 3. Fill in your personal and academic details. 4. Upload mandatory documents (10th/12th marksheets, graduation certificate if applying for PG, photo, ID proof). 5. Pay the application fee and download your receipt.",
    "application": "For application submission: Open the admission portal, create your applicant profile, upload documents in PDF/JPG format (under 2MB), and verify all entries before submitting the application form.",
    "fees": "Tuition fees vary by program (B.Tech, BBA, B.Sc, M.Sc, MCA, MBA) and semester. Payments can be made online via Net Banking, UPI, or Credit/Debit Card through the student portal before the semester deadline. Fee receipts are generated instantly.",
    "placements": "The University Placement Cell organizes campus recruitment drives with leading companies. Students are eligible for placement drives starting in their pre-final/final year after registering with the placement portal.",
    "hostel": "Hostel accommodation is allocated on a first-come, first-served basis. Apply online via the hostel portal after admission confirmation. Facilities include Wi-Fi, 24/7 security, laundry, and dining.",
    "hostel_fees": "Hostel fees depend on the room type (Single, Double, or AC occupancy) and include maintenance. Mess food charges can be paid per semester alongside the accommodation fee.",
    "exam_schedule": "Examination timetables and hall tickets are published 2-3 weeks prior to exams in the Examination Section of the Student Portal. Please check for subject codes, dates, and session timings.",
    "examination": "Semester examinations are conducted at the end of each term. Ensure you meet the minimum 75% attendance requirement and complete exam fee registration to receive your digital admit card.",
    "results": "Semester results and grade cards (SGPA/CGPA) are published on the Examination Portal. Log in with your Student ID/Roll Number to view or download your official marks sheet.",
    "scholarships": "Merit-based, need-based, and government scholarships (National Scholarship Portal, state schemes) are available. Check the Scholarship section on the portal or contact the Dean of Student Welfare office.",
    "library": "The Central Library is open from 8:00 AM to 8:00 PM on weekdays. Students can borrow up to 4 books for 14 days using their university Smart ID card. Digital library access is available 24/7.",
    "library_books": "Search the Online Public Access Catalog (OPAC) using title, author, or ISBN. You can reserve books online and access IEEE, Springer, and ScienceDirect e-journals through the library proxy.",
    "internships": "Internship opportunities are facilitated by the Career Center and department coordinators. Students can also apply for approved industry internships during summer and winter breaks.",
    "contact_information": "For general student assistance, contact the Student Helpdesk at helpdesk@university.edu or call the Admissions Office at +1 (800) 555-0199 (Mon-Fri, 9 AM - 5 PM).",
    "greeting": "Hello! Welcome to Campus Desk. How can I assist you today with UG/PG admissions (B.Tech, BBA, B.Sc, M.Sc, MCA), courses, fees, exams, or campus life?"
}


def build_response_map(df):
    """
    For each intent, keep the SET of unique responses.
    Override canonical responses with rich formatted responses where defined.
    """
    response_map = {}
    for intent, group in df.groupby("intent"):
        top_response = ENHANCED_RESPONSES.get(intent, group["response"].value_counts().idxmax())
        category = group["category"].mode().iloc[0]
        all_resps = sorted(group["response"].unique().tolist())
        if top_response not in all_resps:
            all_resps.append(top_response)
        response_map[intent] = {
            "response": top_response,
            "category": category,
            "all_responses": all_resps
        }
    return response_map



def main():
    t0 = time.time()
    df = load_data()

    # ---- 1. Clean text ----
    df["clean_question"] = df["question"].apply(clean_text)
    df = df[df["clean_question"].str.len() > 0].reset_index(drop=True)

    # ---- 2. Encode labels ----
    le = LabelEncoder()
    y = le.fit_transform(df["intent"])
    X_text = df["clean_question"]

    # ---- 3. Train/test split (stratified) ----
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train_txt)} | Test size: {len(X_test_txt)}")

    # ---- 4. TF-IDF vectorization ----
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)
    print(f"TF-IDF vocab size: {len(vectorizer.vocabulary_)}")

    # ---- 5. Train & compare candidate models ----
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=5, random_state=RANDOM_STATE),
        "LinearSVC": LinearSVC(C=1, random_state=RANDOM_STATE),
        "MultinomialNB": MultinomialNB(),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = []
    trained_models = {}
    for name, clf in candidates.items():
        start = time.time()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        cv_acc = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
        elapsed = time.time() - start
        results.append({
            "model": name, "test_accuracy": round(acc, 4),
            "macro_f1": round(f1, 4), "cv_accuracy_5fold": round(cv_acc, 4),
            "train_time_sec": round(elapsed, 2)
        })
        trained_models[name] = clf
        print(f"{name:20s} | test_acc={acc:.4f} | macro_f1={f1:.4f} | "
              f"cv_acc={cv_acc:.4f} | time={elapsed:.2f}s")

    results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    results_df.to_csv(f"{OUTPUTS_DIR}/model_comparison.csv", index=False)
    print("\nModel comparison:\n", results_df.to_string(index=False))

    # ---- 6. Pick best model ----
    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}")

    # Wrap LinearSVC with calibration to get predict_proba (needed for confidence score)
    if best_name == "LinearSVC":
        best_model = CalibratedClassifierCV(LinearSVC(C=1, random_state=RANDOM_STATE), cv=5)
        best_model.fit(X_train, y_train)

    final_preds = best_model.predict(X_test)
    final_acc = accuracy_score(y_test, final_preds)
    print(f"Final held-out test accuracy: {final_acc:.4f}")

    # ---- 7. Detailed evaluation ----
    report = classification_report(
        y_test, final_preds, target_names=le.classes_, digits=3, output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{OUTPUTS_DIR}/classification_report.csv")

    cm = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {best_name} (Test Accuracy: {final_acc:.2%})")
    plt.xlabel("Predicted Intent")
    plt.ylabel("True Intent")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # Model comparison bar chart
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="test_accuracy", y="model", palette="viridis")
    plt.title("Model Comparison - Test Accuracy")
    plt.xlabel("Accuracy")
    plt.xlim(0, 1)
    for i, v in enumerate(results_df["test_accuracy"]):
        plt.text(v + 0.01, i, f"{v:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/model_comparison.png", dpi=150)
    plt.close()

    # ---- 8. Build response map for retrieval ----
    response_map = build_response_map(df)

    # ---- 9. Save all artifacts ----
    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.pkl")
    joblib.dump(best_model, f"{MODELS_DIR}/intent_classifier.pkl")
    joblib.dump(le, f"{MODELS_DIR}/label_encoder.pkl")
    with open(f"{MODELS_DIR}/response_map.json", "w") as f:
        json.dump(response_map, f, indent=2)
    with open(f"{MODELS_DIR}/metadata.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "test_accuracy": round(final_acc, 4),
            "num_intents": int(len(le.classes_)),
            "intents": le.classes_.tolist(),
            "vocab_size": len(vectorizer.vocabulary_),
            "trained_rows": int(len(df)),
        }, f, indent=2)

    print(f"\nAll artifacts saved to '{MODELS_DIR}/'. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
