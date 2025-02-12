import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs


from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the LSTM model and tokenizer
model = load_model("fake_job_lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_SEQUENCE_LENGTH = 200

def preprocess_text(text):
    # Tokenize and pad the combined text input
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    combined_text = request.form.get("combined_text")

    if not combined_text:
        return render_template("index.html", prediction="Please enter the combined job description text.")

    # Preprocess the input
    input_data = preprocess_text(combined_text)

    # Make prediction
    prediction = model.predict(input_data)[0][0]
    result = "Fraudulent" if prediction > 0.7 else "Legitimate"

    return render_template("index.html", prediction=f"The job post is {result}")
    
if __name__ == "__main__":
    app.run(debug=True)


