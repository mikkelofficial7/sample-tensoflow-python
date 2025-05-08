import numpy as np
import csv
import os
import tensorflow as tf
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
CSV_FILE_PATH_TRUE = "news/True.csv"
CSV_FILE_PATH_FALSE = "news/Fake.csv"
MODEL_PATH = "data_model.keras"
TRAIN_SCRIPT = "trained-data.py"
VOCAB_PATH = "news_vocab.txt"

# Parameters
MAX_LEN = 64  # Adjust to match the sequence length used in training
MAX_TOKENS = 10000  # Set to your desired limit

def run_training_script():
    subprocess.run(['python', TRAIN_SCRIPT])

def load_model_and_vectorizer():
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load vocabulary
    with open(VOCAB_PATH, "r") as f:
        vocab = [line.strip() for line in f.readlines()]

    # Remove duplicates and empty strings
    vocab_formatted = list(set(filter(lambda x: x.strip() != '', vocab)))

    # Recreate the TextVectorization layer
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode='int',
        output_sequence_length=MAX_LEN,
        vocabulary=vocab_formatted
    )

    return model, vectorizer

def cosine_similarity_score(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0]

def compare_with_dataset(user_input, prediction_result):
    true_scores = []
    false_scores = []

    with open(CSV_FILE_PATH_TRUE, 'r') as true_file:
        reader = csv.DictReader(true_file)
        true_samples = [row['text'] for row in list(reader)[:5]]  # Adjust based on CSV structure

    with open(CSV_FILE_PATH_FALSE, 'r') as false_file:
        reader = csv.DictReader(false_file)
        false_samples = [row['text'] for row in list(reader)[:5]]  # Adjust based on CSV structure

    for sample in true_samples:
        score = cosine_similarity_score(user_input, sample)
        true_scores.append(score)

    for sample in false_samples:
        score = cosine_similarity_score(user_input, sample)
        false_scores.append(score)

    if true_scores:
        avg_true_score = sum(true_scores) / len(true_scores)
    else:
        avg_true_score = 0

    if false_scores:
        avg_false_score = sum(false_scores) / len(false_scores)
    else:
        avg_false_score = 0

    print(f"Cosine Prediction Valid: {avg_true_score}")
    print(f"Cosine Prediction Fake: {avg_false_score}")

    result = "Valid News" if avg_true_score >= avg_false_score else "Fake News"
    print(f"Prediction Status: {result}")

def predict_input(model, vectorizer, text_input):
    # Ensure input shape is compatible with model
    text_tensor = tf.constant([text_input])
    processed_input = vectorizer(text_tensor)

    try:
        # Predict
        prediction = model.predict(processed_input)[0][0]

        # Compare with dataset
        compare_with_dataset(text_input, prediction)

    except Exception as e:
        print(f"Prediction error: {str(e)}")

def main():
    # Run training script to ensure the model is ready
    if not os.path.exists(CSV_FILE_PATH_TRUE) or not os.path.exists(CSV_FILE_PATH_FALSE) or not os.path.exists(MODEL_PATH):
        run_training_script()

    model, vectorizer = load_model_and_vectorizer()

    # Loop for user input
    while True:
        user_input = input("Enter news you want to validate (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        predict_input(model, vectorizer, user_input)

if __name__ == "__main__":
    main()
