import os
import cv2
import smtplib
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Load environment variables
load_dotenv()
sender_email = os.getenv("SENDER_EMAIL")
receiver_email = os.getenv("RECEIVER_EMAIL")
email_password = os.getenv("EMAIL_PASSWORD")

# Directories
ALERT_FOLDER = "alert_images"
FRAME_FOLDER = "annotated_frames"
os.makedirs(ALERT_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

def draw_text_with_background(frame, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(frame, (x, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

def send_summary_email(image_paths):
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "PPE Detection Report"

    if image_paths:
        body = "🚨 Safety violations detected! Frames attached where a person was seen without a hardhat."
        for path in image_paths:
            with open(path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                filename = os.path.basename(path)
                part.add_header("Content-Disposition", f"attachment; filename={filename}")
                message.attach(part)
    else:
        body = "✅ Great job! All persons in the video were detected with hardhats. No violations found."

    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, email_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("📧 Email sent successfully.")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")

def process_video(input_path):
    model = YOLO("Model/ppe.pt")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"❌ Could not open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    alert_images = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model(frame)
        person_detected = False
        hardhat_detected = False

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = f"{model.names[cls]} ({box.conf[0]:.2f})"
                    color = (0, 255, 0) if model.names[cls] == "Hardhat" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_text_with_background(frame, label, (x1, y1 - 10))
                    if model.names[cls] == "Person":
                        person_detected = True
                    if model.names[cls] == "Hardhat":
                        hardhat_detected = True

        # Save every processed frame
        frame_filename = f"{FRAME_FOLDER}/frame_{frame_idx:05d}.jpg"
        cv2.imwrite(frame_filename, frame)

        # Save violation frames
        if person_detected and not hardhat_detected:
            alert_filename = f"{ALERT_FOLDER}/alert_{frame_idx:05d}.jpg"
            cv2.imwrite(alert_filename, frame)
            alert_images.append(alert_filename)
            print(f"🚨 Alert saved: {alert_filename}")

    cap.release()
    print(f"✅ Processed {frame_idx} frames.")
    return fps, frame_idx, alert_images

def recreate_video_from_frames(output_path, fps):
    frame_files = sorted(os.listdir(FRAME_FOLDER))
    if not frame_files:
        print("⚠️ No frames found to create video.")
        return

    first_frame = cv2.imread(os.path.join(FRAME_FOLDER, frame_files[0]))
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for file in frame_files:
        frame = cv2.imread(os.path.join(FRAME_FOLDER, file))
        out.write(frame)

    out.release()
    print(f"🎬 Recreated video saved as: {output_path}")

if __name__ == "__main__":
    input_video = "2.mp4"
    output_video = "output_annotated.mp4"

    fps, total_frames, alert_images = process_video(input_video)
    recreate_video_from_frames(output_video, fps)
    send_summary_email(alert_images)
    print("✅ Done.")
