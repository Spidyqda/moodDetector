from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from keras_preprocessing.image import load_img
from keras.models import model_from_json
import numpy as np


app = Flask(__name__)
json_file = open("models/facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("models/facialemotionmodel.h5")

label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


@app.route('/', methods=["GET"])
def hello_world():
    return render_template('index.html')

def preprocess_image(image_path):
    img = load_img(image_path, grayscale=True, target_size=(48, 48))
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

@app.route('/', methods=['POST'])
def predict_emotion():
    if 'imagefile' not in request.files:
        return jsonify({"error": "No se proporcionó ninguna imagen"})

    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return jsonify({"error": "Nombre de archivo vacío"})

    image_path = './images/' +imagefile.filename
    imagefile.save(image_path)
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_label = label[np.argmax(predictions)]
    return jsonify({"emotion": predicted_label})


if __name__ == '__main__':
    app.run(port=3000, debug=True)
