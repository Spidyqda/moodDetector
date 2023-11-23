from flask import Flask, render_template, Response, jsonify, request
import cv2
from keras.models import model_from_json
from keras_preprocessing.image import load_img
import numpy as np

app = Flask(__name__)

json_file = open("C:/Users/mateo/XD/deteccion/facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("C:/Users/mateo/XD/deteccion/facialemotionmodel.h5")

labels = {0: 'enojado', 1: 'disgustado', 2: 'asustado', 3: 'feliz', 4: 'neutral', 5: 'triste', 6: 'sorprendido'}
label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

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
    return jsonify({"message": "Escritura en archivo habilitada"})

def disable_write_to_file():
    global write_to_file_enabled
    write_to_file_enabled = False
    return jsonify({"message": "Escritura en archivo deshabilitada"})

@app.route('/enable_write_to_file', methods=['POST'])
def enable_write():
    return enable_write_to_file()

@app.route('/disable_write_to_file', methods=['POST'])
def disable_write():
    return disable_write_to_file()

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




#Antiguo texto
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