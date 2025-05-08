import tensorflow as tf

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Simple TensorFlow operation
a = tf.constant(5)
b = tf.constant(3)
c = tf.add(a, b)

print(f"Result of TensorFlow operation: {c.numpy()}")