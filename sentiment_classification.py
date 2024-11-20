pip install gensim
pip install keras
pip install tensorflow
pip install Keras-Preprocessing

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow
import keras
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

#import dataset
df=pd.read_csv("/content/drive/MyDrive/Sentiment_Analysis_train1.csv")

df

#load spacy
import spacy
nlp = spacy.load("en_core_web_sm")

# Text Preprocessing
import re

def text_preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if len(word) > 2])
    return text

#Tokenization
df['cleaned_text'] = df['Phrase'].apply(text_preprocess)
df['tokens'] = df['cleaned_text'].apply(lambda text: text.split())
df['tokens'] = df['tokens'].apply(lambda tokens: [token.text for token in nlp(" ".join(tokens))])
print(df['tokens'])

#Stopwords
df['filtered_tokens'] = df['tokens'].apply(lambda tokens: [token for token in tokens if not nlp.vocab[token].is_stop])

print(df['filtered_tokens'])

#lemmatization
df['lemmatized_tokens'] = df['filtered_tokens'].apply(lambda tokens: [token.lemma_ for token in nlp(" ".join(tokens))])

print(df['lemmatized_tokens'])

df['modified_text'] = df['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))
print(df['modified_text'])

#word embeddings
#word2vec
from gensim.models import Word2Vec
sentences = df['lemmatized_tokens'].tolist()
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)  # sg=1 for Skip-gram, sg=0 for CBOW
df['word2vec_vectors'] = df['lemmatized_tokens'].apply(lambda tokens: [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv])
print(df[['modified_text', 'word2vec_vectors']])



#Deep learning
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

max_len = 300  # Set this based on the max sequence length or a fixed limit.
embedding_dim = 100  # Based on the Word2Vec embedding size.

# Pad each list of word vectors to ensure a uniform shape (max_len x embedding_dim)
df['padded_vectors'] = pad_sequences(
    df['word2vec_vectors'],
    maxlen=max_len,
    dtype='float32',
    padding='post',
    truncating='post',
    value=0.0
).tolist()

print(df.columns)

X = np.array(df['padded_vectors'].tolist())
y = np.array(df['Sentiment'])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Initialize a Sequential model
model = Sequential([
    LSTM(128, input_shape=(max_len, embedding_dim)),  # LSTM layer with 128 units
    Dropout(0.5),                                     # Dropout for regularization
    Dense(64, activation='relu'),                     # Dense layer for additional learning
    Dense(1, activation='sigmoid')                    # Output layer for binary classification
])

# Compile the model with optimizer and loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, epochs=10, batch_size=8, validation_split=0.2)

from sklearn.model_selection import train_test_split

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")




#Traditional machine learning
#feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
df_sampled = df.sample(n=10000, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=5, max_features=1600)  # Adjust parameters as needed
tfidf_matrix = tfidf_vectorizer.fit_transform(df_sampled['modified_text'])
print("TF-IDF Matrix shape:", tfidf_matrix.shape)

#Downloaded an excel file how the sparce matrix looks like in TF-IDF
import pandas as pd
tfidf_sparse = tfidf_matrix.toarray()
feature_names = tfidf_vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_sparse, columns=feature_names)
df_tfidf.to_excel('tfidf_matrix.xlsx', index=False)

#Train test the model
from sklearn.model_selection import train_test_split
X=tfidf_matrix
y = df_sampled['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#ML algorithms
'''from sklearn.svm import SVC

# Initialize the model
model=SVC(C=1,gamma=0.1)

# Train the model
model.fit(X_train, y_train)'''

'''from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)'''

from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

model.fit(X_train, y_train)
y_pred = model.predict(X_train)

print('Model accuracy score for train data: {0:0.4f}'. format(accuracy_score(y_train, y_pred)))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)
cm_matrix = pd.DataFrame(data=cm)
plt.title("Confusion matrix of train data")
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='mako')
plt.show()

from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test data
y_pred = model.predict(X_train)

# Calculate accuracy
accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate a detailed classification report
print(classification_report(y_train, y_pred))

model.fit(X_test, y_test)
y_pred = model.predict(X_test)

model.fit(X_test, y_test)
y_pred = model.predict(X_test)
print('Model accuracy score for test data : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm_matrix = pd.DataFrame(data=cm)
plt.title("Confusion matrix of test data")
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='mako')
plt.show()

from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate a detailed classification report
print(classification_report(y_test, y_pred))

y_test

y_pred

