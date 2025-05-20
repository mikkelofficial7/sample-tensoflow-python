import tensorflow as tf

# Load the .keras model
model = tf.keras.models.load_model("data_model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open("betawi_word_model.tflite", "wb") as f:
    f.write(tflite_model)
print("TfLite conversion is complete and saved.")
