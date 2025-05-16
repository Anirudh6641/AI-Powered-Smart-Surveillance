import cv2
import streamlit as st
import torch
import imutils
import time
import smtplib
from email.message import EmailMessage
import numpy as np

# Email configuration - update these with your credentials and SMTP settings.
EMAIL_ADDRESS = "atldps32@gmail.com"      # sender email
EMAIL_PASSWORD = "your_password"                # sender email password
SMTP_SERVER = "smtp.gmail.com"                # e.g., smtp.gmail.com for Gmail
SMTP_PORT = 465                                 # for SSL (or use 587 for TLS if needed)
RECIPIENT_EMAIL = "atldps32@gmail.com"       # recipient email

def send_email_with_image(image, subject="Alert: Human Detected", body="A human has been detected"):
    """
    Encodes the image as JPEG and sends it as an email attachment.
    """
    ret, buf = cv2.imencode('.jpg', image)
    if not ret:
        st.error("Failed to encode image")
        return

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL
    msg.set_content(body)
    msg.add_attachment(buf.tobytes(), maintype='image', subtype='jpeg', filename='capture.jpg')

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
    except Exception as e:
        st.error(f"Failed to send email: {e}")

def main():
    st.title("Smart Webcam Object Detection with YOLOv5 and Email Alert")
    st.write("This application streams your webcam, detects objects using YOLOv5, and sends an email alert when a human is detected.")

    # Load the YOLOv5 model from PyTorch Hub (using the small model for speed)
    st.write("Loading YOLOv5 model (this may take a moment)...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    st.write("Model loaded successfully!")
    
    # Create a placeholder for the video feed in the Streamlit app
    video_placeholder = st.empty()
    stop_button = st.button("Stop Streaming")
    
    # Open the default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return
    
    # Throttle email alerts to one per specified interval (in seconds)
    last_email_time = 0
    email_interval = 60  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame")
            break

        # Resize frame for faster processing
        frame = imutils.resize(frame, width=640)

        # Run YOLOv5 inference on the frame
        results = model(frame)
        annotated_frame = results.render()[0]

        # Check detections for "person"
        detections = results.xyxy[0].cpu().numpy()  # each detection: [x1, y1, x2, y2, confidence, class]
        human_detected = any(model.names[int(cls)] == "person" for *_, conf, cls in detections)
        if human_detected and (time.time() - last_email_time > email_interval):
            send_email_with_image(frame)
            last_email_time = time.time()

        # Convert annotated frame from BGR to RGB for Streamlit display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame, channels="RGB")
        
        if stop_button:
            break

        # Small delay to ensure smooth streaming
        time.sleep(0.1)

    cap.release()

if __name__ == '__main__':
    main()
