import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from openai import OpenAI
import os

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Review Insights Dashboard", layout="wide")

# --- Load Models and Tokenizer ---
@st.cache_resource
def load_sentiment_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("results/sentiment_model")
    model = DistilBertForSequenceClassification.from_pretrained("results/sentiment_model")
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_clustering_pipeline():
    vectorizer = joblib.load("results/tfidf_vectorizer.joblib")
    kmeans = joblib.load("results/kmeans_model.joblib")
    cluster_to_meta = joblib.load("results/cluster_to_meta.joblib")
    return vectorizer, kmeans, cluster_to_meta

# --- Classification Function ---
def classify_reviews(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return [label_map[p] for p in preds]

# --- Clustering Function ---
def assign_meta_category(titles, vectorizer, kmeans, cluster_to_meta):
    X = vectorizer.transform(titles)
    cluster_ids = kmeans.predict(X)
    return [cluster_to_meta[c] for c in cluster_ids]

# --- Summarization Function ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_category_article(meta_cat, df_cat):
    avg_ratings = df_cat.groupby("parent_asin")["rating"].mean().astype(float)
    top3 = avg_ratings.sort_values(ascending=False).head(3).index.tolist()
    worst = avg_ratings.idxmin()

    sections = []
    for asin in top3:
        title = df_cat[df_cat["parent_asin"] == asin]["title"].iloc[0]
        reviews = df_cat[df_cat["parent_asin"] == asin]["text"]
        sample = reviews.sample(n=min(3, len(reviews)), random_state=42)
        reviews_md = "\n".join(f'- "{r}"' for r in sample)
        sections.append(f"Product: {title} (ASIN: {asin})\nReviews:\n{reviews_md}")

    w_title = df_cat[df_cat["parent_asin"] == worst]["title"].iloc[0]
    w_reviews = df_cat[df_cat["parent_asin"] == worst]["text"]
    w_sample = w_reviews.sample(n=min(2, len(w_reviews)), random_state=42)
    w_reviews_md = "\n".join(f'- "{r}"' for r in w_sample)
    sections.append(f"Worst Product: {w_title} (ASIN: {worst})\nReviews:\n{w_reviews_md}")

    prompt = f"""Generate a short blog-style recommendation article for the category: **{meta_cat}**.
Input Data:
{chr(10).join(sections)}

Instructions:
1. Title: “Top Picks for {meta_cat}”
2. List the Top 3 products with their names and key differences.
3. Under each product, bullet-list the Top 2 complaints.
4. Finally, name the Worst Product and explain why to avoid it.
5. Keep length to ~200–250 words.
6. Use a friendly, informative tone.
Produce plain text with headings and bullets.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional product review writer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=600,
        top_p=0.9
    )
    return response.choices[0].message.content.strip()

# --- Main App ---
def main():
    st.title("🛍️ Automated Review Insights Dashboard")
    st.markdown("Upload a CSV with columns: parent_asin, title, text, rating to get started.")

    uploaded = st.file_uploader("Upload reviews CSV", type=["csv"])
    if not uploaded:
        st.info("Awaiting CSV file upload.")
        return

    df = pd.read_csv(uploaded)
    required = {"parent_asin", "title", "text", "rating"}
    if not required.issubset(df.columns):
        st.error(f"CSV must contain columns: {required}")
        return

    # Load models
    tokenizer, sentiment_model = load_sentiment_model()
    vectorizer, kmeans, cluster_to_meta = load_clustering_pipeline()

    # Classification
    st.header("Sentiment Classification")
    df["sentiment"] = classify_reviews(df["text"].tolist(), tokenizer, sentiment_model)
    st.dataframe(df[["parent_asin", "title", "sentiment"]].head())

    # Clustering
    st.header("Product Clustering")
    df["meta_category"] = assign_meta_category(df["title"].tolist(), vectorizer, kmeans, cluster_to_meta)
    st.bar_chart(df["meta_category"].value_counts())

    # Summarization
    st.header("Recommendation Articles by Category")
    for meta_cat in df["meta_category"].dropna().unique():
        with st.expander(f"Generate article for {meta_cat}"):
            df_cat = df[df["meta_category"] == meta_cat]
            article = generate_category_article(meta_cat, df_cat)
            st.markdown(article)

if __name__ == "__main__":
    main()
