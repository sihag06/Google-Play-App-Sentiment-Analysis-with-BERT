# Sentiment Analysis of Google Play Reviews using BERT  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![Jupyter](https://img.shields.io/badge/Jupyter%20Notebook-orange)

This project demonstrates how to build and fine-tune a **BERT-based sentiment analysis model** on Google Play Store app reviews. The model classifies user reviews into three categories: **negative, neutral, or positive**, providing valuable insights for app developers and users.  

---

## 📋 Problem Statement  
The task is **multi-class text classification** of user reviews.  
Using a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model, we aim to accurately capture the sentiment expressed in textual feedback.  

---

## 💾 Dataset  
- **Source**: Downloaded directly from Google Drive within the notebook.  
- **File**: `reviews.csv`  
- **Columns**:  
  - `content`: The raw text of the review.  
  - `score`: Star rating (1–5) given by the user.  

### Sentiment Mapping  
To address **class imbalance** in star ratings:  
- **Negative (0)** → Ratings **1 or 2**  
- **Neutral (1)** → Rating **3**  
- **Positive (2)** → Ratings **4 or 5**  

---

## ⚙️ Methodology  

### 1. Data Preprocessing  
- Used **BertTokenizer** (`bert-base-cased`)  
- Max sequence length: **160**  
- Applied:  
  - Special tokens (`[CLS]`, `[SEP]`)  
  - Padding & attention masks  

### 2. Model Architecture  
Custom **SentimentClassifier** built on **PyTorch**:  
- **Base Model**: `bert-base-cased` (12 transformer layers, 768-dim output).  
- **Classification Head**:  
  - Dropout (p=0.3)  
  - Linear layer → 3 output classes  

### 3. Training Setup  
- **Optimizer**: AdamW  
- **Scheduler**: Linear LR scheduler  
- **Loss Function**: CrossEntropyLoss  
- **Epochs**: 10  
- **Batch Size**: 16  
- **Split**: Train (90%) / Validation (5%) / Test (5%)  

### 4. Evaluation  
- Metrics: Accuracy, Precision, Recall, F1-score  
- Tools: Scikit-learn classification report & confusion matrix  

---

## 🛠️ Technologies Used  
- **Framework**: PyTorch  
- **NLP**: Hugging Face Transformers  
- **Data Handling**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  
- **Utilities**: Scikit-learn, gdown, watermark  

---

## 📊 Results  

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.89      | 0.87   | 0.88     | 245     |
| Neutral   | 0.83      | 0.85   | 0.84     | 254     |
| Positive  | 0.92      | 0.93   | 0.92     | 289     |
| **Overall** | **0.88** | **0.88** | **0.88** | **788** |

- **Final Test Accuracy**: **88.3%**  
- **Observation**: Model performs strongly on **positive** and **negative** reviews. Slightly lower accuracy on **neutral** reviews (often misclassified as positive/negative).  

### Confusion Matrix  
Add your confusion matrix image here:  
```
![Confusion Matrix](confusion_matrix.png)
```

---

## 🚀 How to Run  

1. Clone the repository:  
   ```bash
   git clone <repo-link>
   cd sentiment-analysis-bert
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:  
   ```bash
   jupyter notebook Sentiment_Analysis_BERT.ipynb
   ```

---
