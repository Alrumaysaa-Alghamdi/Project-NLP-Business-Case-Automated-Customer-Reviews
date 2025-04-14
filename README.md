# Project-NLP-Business-Case-Automated-Customer-Reviews
This project leverages Natural Language Processing (NLP) and clustering techniques to help business owners analyze large-scale customer reviews automatically. It consists of three key components:

---

## 🔹 1. **Review Sentiment Classification**
### **Objective**:
Classify customer reviews into three sentiment categories: **Positive**, **Neutral**, and **Negative** based on their textual content.

### **What We Did**:
- Used a pre-trained transformer model (**DistilBERT**) from Hugging Face.
- Fine-tuned the model on the `Amazon-Reviews-2023` dataset, using review texts and mapped star ratings.
- Mapped star ratings to sentiment:
  - 1–2 stars → Negative
  - 3 stars   → Neutral
  - 4–5 stars → Positive
- Applied class weights during training to handle class imbalance.
- Evaluated performance using accuracy, precision, recall, F1-score, and confusion matrix.
- Saved the trained model to be used in the web app.

---

## 🔹 2. **Product Category Clustering**
### **Objective**:
Group products into broader meta-categories based on their titles, simplifying review analysis and recommendation generation.

### **What We Did**:
- Extracted product titles from metadata.
- Transformed titles using **TF-IDF vectorization**.
- Applied **K-Means clustering** and tuned the number of clusters using:
  - **WCSS and Elbow Method**
- Interpreted top terms in each cluster to assign meaningful names.
- Saved the vectorizer, KMeans model, and cluster-to-meta-category mapping.

---

## 🔹 3. **Review Summarization with Generative AI**
### **Objective**:
Generate blog-style recommendation articles for each product category based on customer reviews.

### **What We Did**:
- Used GPT-3 from OpenAI to generate natural summaries.
- For each cluster/category:
  - Identified the **top 3 products** (by average rating).
  - Identified the **worst-rated product**.
  - Sampled reviews to highlight top complaints.
- Crafted prompts with structured instructions to guide GPT.
- Generated summaries that include product highlights, key differences, and warnings.

---

## 🌐 Web Application (Streamlit)
Built an interactive dashboard where users can:
- Upload a CSV file of reviews.
- Classify sentiment of reviews using the trained DistilBERT model.
- View clustering and top terms.
- Generate personalized recommendation articles for each cluster.

---

## 📊 Dataset Used
We used the **Amazon Reviews 2023** dataset provided by the McAuley Lab, which contains large-scale customer review data, metadata, and rating records for Amazon products across multiple categories.

**Source**: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

**Citation**:
```
@article{ni2019justifying,
  title={Justifying recommendations using distantly-labeled reviews and fine-grained aspects},
  author={Ni, Jianmo and Li, Jiacheng and McAuley, Julian},
  journal={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2019}
}
```

> 💡 This project transforms raw customer reviews into business-friendly insights using powerful NLP and clustering techniques.
