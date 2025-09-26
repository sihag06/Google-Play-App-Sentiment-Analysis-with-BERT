#Sentiment Analysis of Google Play Reviews using BERT
This project demonstrates how to build and train a sophisticated sentiment analysis model using the BERT (Bidirectional Encoder Representations from Transformers) architecture. The model is fine-tuned to classify user reviews from the Google Play Store into three categories: negative, neutral, or positive.

📋 Problem Statement
The goal of this project is to perform multi-class text classification on Google Play app reviews. By leveraging a pre-trained BERT model, we aim to accurately categorize the sentiment expressed in user feedback, which can provide valuable insights for app developers and users.

💾 Dataset
The dataset used in this project consists of Google Play app reviews.

Source: The data is downloaded directly from Google Drive within the notebook.

Content: The primary file, reviews.csv, contains several columns, but the analysis focuses on:

content: The raw text of the user review.

score: The star rating (1-5) given by the user.

The initial distribution of scores was highly imbalanced, with a majority of 5-star ratings. To create a more balanced dataset for training, these scores were mapped to three sentiment classes.

⚙️ Methodology
The project follows a structured approach to building the sentiment analysis model.

1. Data Preprocessing
Sentiment Mapping: The 1-5 star ratings were converted into three sentiment categories to address class imbalance:

Negative (0): Ratings of 1 or 2.

Neutral (1): Rating of 3.

Positive (2): Ratings of 4 or 5.

Tokenization: The text data was processed using the BertTokenizer from the Hugging Face library, specifically for the bert-base-cased model. A maximum sequence length of 160 was chosen based on an analysis of review lengths. The tokenizer handles:

Adding special tokens ([CLS], [SEP]).

Padding sequences to a uniform length.

Creating an attention mask to differentiate real tokens from padding.

2. Model Architecture
A custom SentimentClassifier was built in PyTorch by fine-tuning the pre-trained BertModel.

The architecture consists of:

BERT Base: The bert-base-cased model, which acts as the core feature extractor. It has 12 transformer layers and produces a 768-dimensional output vector (pooled_output) that summarizes the input text.

Custom Head:

A Dropout layer (p=0.3) for regularization to prevent overfitting.

A final Linear layer that maps the 768-dimensional BERT output to the 3 sentiment classes.

3. Training and Evaluation
Setup: The model was trained for 10 epochs using the AdamW optimizer and a linear learning rate scheduler. The loss function used was CrossEntropyLoss.

Data Handling: The data was split into training (90%), validation (5%), and test (5%) sets. PyTorch DataLoader objects were used to efficiently feed data in batches of 16.

Evaluation: The model's performance was evaluated based on accuracy on the validation set during training. The final performance was measured on the unseen test set.

🛠️ Technologies Used
Framework: PyTorch

NLP Library: Hugging Face Transformers

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Utilities: Scikit-learn, gdown, watermark

📊 Results
The fine-tuned BERT model achieved excellent performance on the test set.
| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.89      | 0.87   | 0.88     | 245     |
| Neutral   | 0.83      | 0.85   | 0.84     | 254     |
| Positive  | 0.92      | 0.93   | 0.92     | 289     |
| **Overall**   | **0.88**      | **0.88**   | **0.88**     | **788**     |

Final Test Accuracy: 88.3%
The classification report provides a more detailed breakdown:
The confusion matrix below highlights the model's performance. It shows high accuracy for positive and negative reviews but reveals a slight difficulty in classifying neutral reviews, which are sometimes mistaken for positive or negative.
