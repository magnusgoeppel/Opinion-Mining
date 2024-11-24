import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Download the Sentiment140 dataset
path = kagglehub.dataset_download("kazanova/sentiment140")

# Load the dataset
dataset_path = f"{path}/training.1600000.processed.noemoticon.csv"
cols = ['target', 'id', 'date', 'flag', 'user', 'text']
df = pd.read_csv(dataset_path, encoding='ISO-8859-1', header=None, names=cols)

# Use only the required columns
df = df[['target', 'text']]

# Proceed with your analysis
df = df[df['target'].isin([0, 4])]
df['target'] = df['target'].replace({0: 'negative', 4: 'positive'})

samplesize = 100000
df_sampled = df.groupby('target').apply(lambda x: x.sample(n=samplesize, random_state=42)).reset_index(drop=True)
# Evaluate distribution
print(df_sampled['target'].value_counts())
print(df_sampled.head())

X = df_sampled['text']
Y = df_sampled['target']
# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=df_sampled['target'], random_state=42)

# TF-IDF for Naive Bayes
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Training Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_predictions = nb_model.predict(X_test_tfidf)

# Naive Bayes Evaluation
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_report = classification_report(y_test, nb_predictions, output_dict=True)
nb_cm = confusion_matrix(y_test, nb_predictions)

print("Naive Bayes Accuracy:", nb_accuracy)
print(classification_report(y_test, nb_predictions))

# Tokenisation and Padding for LSTM
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

# Create LSTM Model
lstm_model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(units=64, return_sequences=False),
    Dense(1, activation='sigmoid')  # Nur eine Ausgabe (sigmoid für binäre Klassifikation)
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

y_train_lstm = y_train.replace({'negative': 0, 'positive': 1}).values
y_test_lstm = y_test.replace({'negative': 0, 'positive': 1}).values

# Train LSTM Model
lstm_model.fit(X_train_seq, y_train_lstm, epochs=16, validation_data=(X_test_seq, y_test_lstm), batch_size=256)

# LSTM Model evaluation
lstm_predictions = lstm_model.predict(X_test_seq)
lstm_predictions_classes = (lstm_predictions > 0.5).astype("int32").flatten()
lstm_cm = confusion_matrix(y_test_lstm, lstm_predictions_classes)

# LSTM Model print Results
print("LSTM Accuracy:", accuracy_score(y_test_lstm, lstm_predictions_classes))
print(classification_report(y_test_lstm, lstm_predictions_classes))

# Compare the Models
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot confusion matrix for Bayes
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=ax[0])
ax[0].set_title('Confusion Matrix - Naive Bayes Modell', fontsize=16)
ax[0].set_xlabel('Predicted', fontsize=12)
ax[0].set_ylabel('Actual', fontsize=12)

# Plot confusion matrix for LSTM
sns.heatmap(lstm_cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=ax[1])
ax[1].set_title('Confusion Matrix - LSTM', fontsize=16)
ax[1].set_xlabel('Predicted', fontsize=12)
ax[1].set_ylabel('Actual', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()