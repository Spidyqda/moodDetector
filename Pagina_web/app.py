from flask import Flask, render_template, Response, jsonify, request
import cv2
from keras.models import model_from_json
from keras_preprocessing.image import load_img
import numpy as np
import os

app = Flask(__name__)

#Carga del modelo
json_file = open("model/facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model/facialemotionmodel.h5")


#Creacion de etiquetas
labels = {0: 'enojado', 1: 'disgustado', 2: 'asustado', 3: 'feliz', 4: 'neutral', 5: 'triste', 6: 'sorprendido'}
label = ['enojado', 'disgustado', 'asustado', 'feliz', 'neutral', 'triste', 'sorprendido']
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

#ENDPOINTS

@app.route('/enable_write_to_file', methods=['POST'])
def enable_write():
    return enable_write_to_file()

@app.route('/disable_write_to_file', methods=['POST'])
def disable_write():
    return disable_write_to_file()

@app.route('/get_emotion_counts', methods=['GET'])
def get_emotion_counts():
    emotion_counts = count_emotions_from_file()
    return jsonify(emotion_counts)

@app.route('/get_results')
def get_results():
    emotion_counts = count_emotions_from_file()
    scale_letter = assign_scale(emotion_counts)
    results_text = f"Escala: {scale_letter}\nEmociones: {str(emotion_counts)}"
    return results_text

@app.route('/', methods=['POST'])
def predict_emotion():
    if 'imagefile' not in request.files:
        return f"No se proporcionó ninguna imagen"

    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return f"Nombre de archivo vacío"

    image_path = './images/' +imagefile.filename
    imagefile.save(image_path)
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_label = label[np.argmax(predictions)]
    return predicted_label


#FUNCIONES

def preprocess_image(image_path):
    img = load_img(image_path, grayscale=True, target_size=(48, 48))
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def write_results_to_file(results):
    if write_to_file_enabled:
        with open("results.txt", "a") as file:
            file.write(results + "\n")


write_to_file_enabled = True

def enable_write_to_file():
    global write_to_file_enabled
    write_to_file_enabled = True
    if os.path.exists("results.txt"):
        os.remove("results.txt")
    return jsonify({"message": "Prueba de deteccion iniciada"})

def disable_write_to_file():
    global write_to_file_enabled
    write_to_file_enabled = False
    return jsonify({"message": "Prueba de deteccion terminada"})

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 0, 0), 2)
            cv2.putText(frame, prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

            results = f"Predicted emotion: {prediction_label}, Coordinates: ({p}, {q}, {r}, {s})"
            write_results_to_file(results)
    except cv2.error:
        pass

def generate_frames():
    webcam = cv2.VideoCapture(0)

    while True:
        success, frame = webcam.read()
        if not success:
            break
        else:
            detect_emotion(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def count_emotions_from_file():
    emotion_counts = {"enojado": 0, "disgustado": 0, "asustado": 0, "feliz": 0, "neutral": 0, "triste": 0, "sorprendido": 0}
    with open("results.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            emotion = line.split(",")[0].split(":")[1].strip()
            emotion_counts[emotion] += 1
    return emotion_counts

#Escala de calificacion
def assign_scale(emotion_counts):
    max_emotion = max(emotion_counts, key=emotion_counts.get)
    if max_emotion in ["feliz", "sorprendido"]:
        return "A"
    elif max_emotion == "neutral":
        return "B"
    elif max_emotion == "triste":
        return "C"
    elif max_emotion in ["asustado", "disgustado"]:
        return "D"
    elif max_emotion == "enojado":
        return "F"
    else:
        return "No se detectaron resultados"


#Redireccionamientos
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camara/')
def camara():
    return render_template('camara.html')

@app.route('/demostracion/')
def demostracion():
    return render_template('demostracion.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(port=3000, debug=True)