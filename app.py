from flask import Flask, render_template, Response, jsonify, url_for
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import random

app = Flask(__name__)

# Load pre-trained pedestrian detection model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Generate a pedestrian heatmap
def generate_heatmap():
    x = np.random.randint(0, 100, 500)
    y = np.random.randint(0, 100, 500)
    
    plt.figure(figsize=(6, 5))
    sns.kdeplot(x=x, y=y, cmap="Reds", fill=True, bw_adjust=0.5)
    plt.title("Pedestrian Density Heatmap")

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return img_str

# AI-powered pedestrian detection from video
def generate_frames():
    cap = cv2.VideoCapture("pedestrian_video.mp4")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Detect pedestrians
        boxes, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# IoT Sensor API Simulation
@app.route("/sensor-activate")
def sensor_activate():
    status = random.choice(["Motion Detected! Escalator Activated", "No Motion Detected"])
    return jsonify({"status": status})

# Emergency Alert API
@app.route("/trigger-alert")
def trigger_alert():
    return jsonify({"alert": "Emergency Alert Sent to Authorities!"})

# Home Page
@app.route("/")
def index():
    heatmap_img = generate_heatmap()
    return render_template("index.html", heatmap_img=heatmap_img)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
