import tensorflow as tf

# Load your existing model
print("Loading model...")
model = tf.keras.models.load_model("fake_job_lstm_model.h5")
print("Model loaded successfully!")

# Convert the model to TFLite format
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Allow TensorFlow ops that are not natively supported by TFLite
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Default TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow ops that are not natively supported
]

# Fix the TensorListReserve issue
converter._experimental_lower_tensor_list_ops = False  # Keep TensorList ops

# Fix variable constant folding issue
converter.experimental_enable_resource_variables = True

# Convert and save the model
tflite_model = converter.convert()
with open("fake_job_lstm_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite successfully! ✅")
