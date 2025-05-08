import numpy as np
import csv
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import subprocess

# Paths
CSV_FILE_PATH = 'sample_dataset.csv'
MODEL_PATH = "data_model.keras"
TOKENIZER_PATH = "model_tokenizer.json"
TRAIN_SCRIPT = "trained-data.py"

# Parameters
MAX_LEN = 50  # Adjust based on your model's input length

def run_training_script():
    """Run the training script."""
    subprocess.run(['python', TRAIN_SCRIPT])

def load_model_and_tokenizer():
    """Load the trained model and tokenizer."""
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(TOKENIZER_PATH, "r") as f:
        tokenizer_data = f.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)
    
    return model, tokenizer

def write_to_csv(text, label, prediction):
    """Write the prediction to the CSV file."""
    file_exists = os.path.isfile(CSV_FILE_PATH)
    
    # Prepare data for writing
    data = [text, label, prediction]  # Ensure prediction is a float, not an array
    
    with open(CSV_FILE_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")  # Use semicolon as separator

        if not file_exists:
            writer.writerow(["text", "label", "prediction"])  # Write header only once
        writer.writerow(data)

    # Re-run training to update model with new data
    run_training_script()

def predict_and_save(model, tokenizer, text):
    """Predict the sentiment and save the result to the CSV."""
    # Prepare input data
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

    # Predict sentiment
    prediction = model.predict(padded)[0][0]
    # Determine class label (0 or 1)
    label = 1 if prediction > 0.55555 else 0
    
    print(f"Text: {text}\nPrediction: {'Positive' if label == 1 else 'Negative'} ({prediction:.5f})\n")
    
    # Save to CSV
    write_to_csv(text, label, prediction)

def main():
    # Run training script to ensure the model is ready
    if not os.path.exists(CSV_FILE_PATH) or not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        run_training_script()

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Loop for user input
    while True:
        user_input = input("Enter text to predict (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        predict_and_save(model, tokenizer, user_input)

if __name__ == "__main__":
    main()
