import os
import cv2
import time
import smtplib
import streamlit as st
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tempfile

# Load .env credentials
load_dotenv()
sender_email = os.getenv("SENDER_EMAIL")
receiver_email = os.getenv("RECEIVER_EMAIL")
email_password = os.getenv("EMAIL_PASSWORD")

# Folder for violator images
VIOLATION_DIR = "violator_images"
os.makedirs(VIOLATION_DIR, exist_ok=True)

def clear_violator_folder():
    for f in os.listdir(VIOLATION_DIR):
        os.remove(os.path.join(VIOLATION_DIR, f))

def draw_text_with_background(frame, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    x, y = position
    cv2.rectangle(frame, (x, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness)

def overlay_dashboard(frame, metrics):
    x, y = 10, 20
    spacing = 20
    for key, value in metrics.items():
        draw_text_with_background(frame, f"{key}: {value}", (x, y))
        y += spacing
    return frame

def send_summary_email(folder, violations, total_frames, summary_text):
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "PPE Violation Summary"

    imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]
    message.attach(MIMEText(summary_text, "plain"))

    for img in imgs:
        with open(img, "rb") as file:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(img)}")
            message.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, email_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("Email sent.")
    except Exception as e:
        print("Email error:", e)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def detect_and_process(source, is_live=False, thresholds=None):
    clear_violator_folder()

    model = YOLO("Model/ppe.pt")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Cannot open video/camera.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_idx = 0
    alert_count = 0
    last_capture_time = time.time()

    stframe = st.empty()

    detections = {
        "Person": [],
        "Hardhat": [],
        "Face Mask": [],
        "Safety Vest": []
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        result = model(frame, verbose=False)[0]
        boxes = result.boxes

        detections = {k: [] for k in detections}

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                if label in thresholds and conf >= thresholds[label]:
                    detections[label].append((x1, y1, x2, y2))
                    color = (0, 255, 0) if label != "Person" else (255, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_text_with_background(frame, f"{label} ({conf:.2f})", (x1, y1 - 10))

        for (px1, py1, px2, py2) in detections["Person"]:
            person_box = (px1, py1, px2, py2)
            has_helmet = any(iou(person_box, (hx1, hy1, hx2, hy2)) > 0.2
                             for (hx1, hy1, hx2, hy2) in detections["Hardhat"])
            if not has_helmet:
                crop = frame[py1:py2, px1:px2]
                crop = cv2.resize(crop, (150, 200))
                now = time.time()
                if now - last_capture_time > 8:
                    cv2.imwrite(f"{VIOLATION_DIR}/violation_{frame_idx}.jpg", crop)
                    last_capture_time = now
                alert_count += 1
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                draw_text_with_background(frame, "Person", (px1, py1 - 10))

        metrics = {
            "Persons": len(detections["Person"]),
            "Hardhats": len(detections["Hardhat"]),
            "Face Masks": len(detections["Face Mask"]),
            "Vests": len(detections["Safety Vest"]),
            "Violations": alert_count
        }

        frame = overlay_dashboard(frame, metrics)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Frame {frame_idx}")

        if is_live and cv2.waitKey(1) == 27:
            break

    cap.release()

    total_persons = len(detections["Person"])
    total_hardhats = len(detections["Hardhat"])
    total_masks = len(detections["Face Mask"])
    total_vests = len(detections["Safety Vest"])

    # Ensure number of persons is always more than or equal to other equipment
    total_hardhats = min(total_hardhats, total_persons)
    total_masks = min(total_masks, total_persons)
    total_vests = min(total_vests, total_persons)

    extra_violation_flag = False
    violation_details = ""

    if total_persons > total_hardhats:
        extra_violation_flag = True
        violation_details += f"- {total_persons} persons, only {total_hardhats} hardhats\n"
    if total_persons > total_masks:
        extra_violation_flag = True
        violation_details += f"- {total_persons} persons, only {total_masks} face masks\n"
    if total_persons > total_vests:
        extra_violation_flag = True
        violation_details += f"- {total_persons} persons, only {total_vests} vests\n"

    summary_msg = f"""
Summary Report:
Persons: {total_persons}
Hardhats: {total_hardhats}
Face Masks: {total_masks}
Vests: {total_vests}
Violation Frames: {alert_count}
"""
    if extra_violation_flag:
        summary_msg += "\nAdditional PPE Violations Detected:\n" + violation_details

    send_summary_email(VIOLATION_DIR, alert_count, frame_idx, summary_msg)
    st.success(f"Processed {frame_idx} frames with {alert_count} violations.")
    st.success("Email sent with violator snapshots.")

st.set_page_config("PPE Detection Dashboard", layout="wide")
st.title("PPE Detection Dashboard")

st.sidebar.title("Detection Confidence Thresholds")
thresholds = {
    "Person": st.sidebar.slider("Person", 0.0, 1.0, 0.5, 0.05),
    "Hardhat": st.sidebar.slider("Hardhat", 0.0, 1.0, 0.5, 0.05),
    "Face Mask": st.sidebar.slider("Face Mask", 0.0, 1.0, 0.5, 0.05),
    "Safety Vest": st.sidebar.slider("Safety Vest", 0.0, 1.0, 0.5, 0.05),
}

mode = st.radio("Select Input", ["Upload Video", "Live Camera"])

if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        detect_and_process(tfile.name, is_live=False, thresholds=thresholds)

elif mode == "Live Camera":
    if st.button("Start Live Detection"):
        detect_and_process(0, is_live=True, thresholds=thresholds)
