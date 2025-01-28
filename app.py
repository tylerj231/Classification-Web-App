import os

from flask import Flask, request, render_template

import tensorflow as tf

from classifier import classify

app = Flask(__name__)

STATIC_FOLDER = 'static'
UPLOAD_FOLDER = "static/uploads/"

cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "cats_and_dogs.keras")
@app.route('/')
def home():
    return render_template('index.html')


@app.post("/classify")
def upload_file():
    file = request.files["image"]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_image_path)

    label, prob = classify(cnn_model, upload_image_path)
    prob = max(0, min(prob, 1))
    confidence = prob * 100

    return {
        "label": label,
        "probability": round(confidence, 2),
    }

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
