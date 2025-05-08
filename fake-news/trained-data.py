import os
import pandas as pd
import numpy as np
import tensorflow as tf

MAX_TOKENS = 10000
OUTPUT_DIM = 16
OUTPUT_SEQUENCE_LENGTH = 64

csv_file_path_true = "news/True.csv"
csv_file_path_false = "news/Fake.csv"

df_true = pd.read_csv(csv_file_path_true)
df_false = pd.read_csv(csv_file_path_false)

# Add labels
df_true["label"] = 1
df_false["label"] = 0

# Combine the datasets
df_combined = pd.concat([df_true, df_false]).reset_index(drop=True)

# Extract data
titles = df_combined['title'].astype(str).values
texts = df_combined['text'].astype(str).values
subjects = df_combined['subject'].astype(str).values
dates = df_combined['date'].astype(str).values
labels = df_combined['label'].values

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode='int',
    output_sequence_length=OUTPUT_SEQUENCE_LENGTH
)

# Adapt the vectorization layer
all_text = tf.concat([titles, texts, subjects, dates], axis=0)
vectorize_layer.adapt(all_text)

# Save the vocabulary to a file
vocab_path = "news_vocab.txt"
with open(vocab_path, "w") as f:
    for token in vectorize_layer.get_vocabulary():
        f.write(f"{token}\n")

def preprocess(title, text, subject, date, label):
    # Combine strings into a single tensor
    combined = tf.strings.join([title, text, subject, date], separator=" | ")
    
    # Vectorize the combined text
    vectorized = vectorize_layer(combined)
    
    return vectorized, label
    
# Create the dataset
dataset = tf.data.Dataset.from_tensor_slices((titles, texts, subjects, dates, labels))
dataset = dataset.map(preprocess).shuffle(100).batch(32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(OUTPUT_SEQUENCE_LENGTH,)),  # 64 tokens after vectorization
    tf.keras.layers.Embedding(input_dim=MAX_TOKENS, output_dim=OUTPUT_DIM),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(dataset, epochs=5)

# Save the model
model.save("data_model.keras")
print("Model training complete and saved.")
