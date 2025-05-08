import os
import pandas as pd
import numpy as np
import tensorflow as tf
import subprocess
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load the dataset
csv_file_path = "sample_dataset.csv"

df = pd.read_csv(csv_file_path, sep=';')

# Extract texts and labels
texts = df['text'].values
labels = df['label'].values
prediction = df['prediction'].values

# Tokenizer configuration
max_words = 1000
max_len = 50
oov_token = "<OOV>"

# Tokenize the text
tokenizer = Tokenizer(num_words=max_words, oov_token=oov_token)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Label distribution check
unique, counts = np.unique(labels, return_counts=True)
print(f"Label distribution: {dict(zip(unique, counts))}")

# Save the tokenizer for later use
tokenizer_json = tokenizer.to_json()
with open("model_tokenizer.json", "w") as f:
    f.write(tokenizer_json)

# Split data
X_train, X_test, y_train, y_test, pred_train, pred_test = train_test_split(padded_sequences, labels, prediction, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
print(f"Training label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(X_train, np.array(y_train), epochs=5, batch_size=4, validation_data=(X_test, np.array(y_test)))

# Save the model
model.save("data_model.keras")
print("Model training complete and saved.")
