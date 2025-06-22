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
SUMMARY_FOLDER = "highlighted_summary"
os.makedirs(ALERT_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

def draw_text_with_background(frame, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(frame, (x, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

def send_summary_email(summary_folder, violations, total_frames):
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "PPE Detection Alert Report"

    summary_images = [os.path.join(summary_folder, f)
                      for f in os.listdir(summary_folder)
                      if f.lower().endswith(".jpg")]

    if violations > 0:
        body = f"""🚨 There are {violations} frames with rule violation (person without hardhat).
Attached are summary snapshots taken every 80 frames with only the hardhat issues highlighted."""
    else:
        body = f"✅ All {total_frames} frames processed. No violations detected."

    message.attach(MIMEText(body, "plain"))

    for path in summary_images:
        with open(path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            filename = os.path.basename(path)
            part.add_header("Content-Disposition", f"attachment; filename={filename}")
            message.attach(part)

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
        return None, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    alert_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model(frame)
        person_count = 0
        hardhat_count = 0
        violations_to_draw = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    conf = float(box.conf[0])
                    color = (0, 255, 0) if label == "Hardhat" else (0, 0, 255)

                    if label == "Person":
                        person_count += 1
                    elif label == "Hardhat":
                        hardhat_count += 1

                    # Draw everything
                    text = f"{label} ({conf:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_text_with_background(frame, text, (x1, y1 - 10))

                    # Track missing hardhat only
                    if label in ["Person"]:  # if no Hardhat detected later
                        violations_to_draw.append((x1, y1, x2, y2))

        # Summary text
        y_offset = 30
        draw_text_with_background(frame, f"Persons: {person_count}", (10, y_offset))
        draw_text_with_background(frame, f"Hardhats: {hardhat_count}", (10, y_offset + 25))

        # Save main annotated frame
        frame_filename = f"{FRAME_FOLDER}/frame_{frame_idx:05d}.jpg"
        cv2.imwrite(frame_filename, frame)

        # Save violation frames
        if person_count > 0 and hardhat_count < person_count:
            alert_filename = f"{ALERT_FOLDER}/alert_{frame_idx:05d}.jpg"
            cv2.imwrite(alert_filename, frame)
            alert_count += 1
            print(f"🚨 Alert saved: {alert_filename}")

            # Save summary every 80 frames
            if frame_idx % 70 == 0:
                summary = frame.copy()
                for (x1, y1, x2, y2) in violations_to_draw:
                    cv2.rectangle(summary, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    draw_text_with_background(summary, "No Hardhat", (x1, y1 - 10))
                summary_path = f"{SUMMARY_FOLDER}/summary_{frame_idx:05d}.jpg"
                cv2.imwrite(summary_path, summary)
                print(f"📝 Summary frame saved: {summary_path}")

    cap.release()
    print(f"✅ Processed {frame_idx} frames. Violations: {alert_count}")
    return fps, frame_idx, alert_count

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

    fps, total_frames, alert_count = process_video(input_video)

    if fps is not None:
        recreate_video_from_frames(output_video, fps)
        send_summary_email(SUMMARY_FOLDER, alert_count, total_frames)

    print("✅ All steps completed.")
