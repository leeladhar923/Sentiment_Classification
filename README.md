# Sentiment_Classification
Sentiment Classification using Spacy, Deep learning and traditional machine learning models.

**Sentiment Classification using Traditional Machine Learning and Deep Learning**

This project performs sentiment classification using text data. It integrates text preprocessing, word embeddings, deep learning, and traditional machine learning models for comparison. The results include accuracy metrics, confusion matrices, and detailed classification reports.

**Features:**
**Text Preprocessing:** Tokenization, lemmatization, and stopword removal using SpaCy.
**Word Embeddings:** Using Word2Vec for dense vector representations.
**Deep Learning:** Sentiment classification with an LSTM-based model.
**Traditional Machine Learning:** TF-IDF feature extraction and classification with algorithms like Random Forest.
**Visualization:** Confusion matrices with heatmaps for performance evaluation.

**Usage:**

**1.Import and Preprocess Data**

Load dataset.
Clean text by removing special characters, digits, and short words.
Tokenize, remove stopwords, and lemmatize using SpaCy.

**2.Generate Word Embeddings**

Use Word2Vec to generate word embeddings for each phrase.

**3.Deep Learning Model**

Create padded sequences from word embeddings.
Train an LSTM model for sentiment classification.

**4.Traditional Machine Learning**

Extract features using TF-IDF.
Train and evaluate models like Random Forest.

**5.Evaluation**

Calculate accuracy, classification reports, and visualize confusion matrices.
