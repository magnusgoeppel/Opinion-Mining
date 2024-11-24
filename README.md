# Opinion-Mining

This repository contains a Python-based sentiment analysis project that classifies text data into **positive** or **negative** categories. It utilizes two different machine learning approaches:

1. **Naive Bayes Model**: A probabilistic method with TF-IDF vectorization.
2. **LSTM Model**: A deep learning model using word embeddings and sequential data processing.

## Dataset
### **Sentiment140 Dataset**
The **Sentiment140 dataset** contains 1.6 million tweets with sentiment labels:
- `0`: Negative sentiment
- `4`: Positive sentiment

For this project:
- Tweets are sampled to create a balanced dataset of 100,000 samples per sentiment class.
- Only the `target` (sentiment) and `text` columns are used for the analysis.

## Project Workflow

1. **Data Preprocessing**:
    - Load and filter data to keep only `negative` and `positive` classes.
    - Convert numerical labels into categorical labels (`negative` or `positive`).
    - Split data into training and testing sets (80/20 split).

2. **Model 1: Naive Bayes with TF-IDF**:
    - Preprocess text with TF-IDF vectorization.
    - Train a Multinomial Naive Bayes model.
    - Evaluate model performance using accuracy, classification report, and confusion matrix.

3. **Model 2: LSTM with Word Embeddings**:
    - Tokenize text data and convert it to sequences.
    - Pad sequences to a fixed length of 100.
    - Train an LSTM model with:
        - Embedding layer
        - LSTM layer
        - Dense output layer with sigmoid activation.
    - Evaluate model performance using accuracy, classification report, and confusion matrix.

4. **Results Visualization**:
    - Confusion matrices are plotted for both models to compare performance.

## Results

### Naive Bayes Model
- **Accuracy**: ~74.6%
- Strengths: Faster training and inference, interpretable.
- Weaknesses: Relies on simple linear relationships, struggles with complex patterns in text.

### LSTM Model
- **Accuracy**: ~75.8%
- Strengths: Captures contextual and sequential patterns in text.
- Weaknesses: Longer training time, computationally expensive.
