import cv2
import numpy as np
import os
from datetime import datetime

def nothing(x):
    pass

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:-1]]
        idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


# === CONFIG ===
video_path = r"C:\Users\dhana\Downloads\Planetary_Gear_Detection\Planetary_Gear_Backup.mp4" # 🔁 Change this to your video path
template_folder = r"C:\Users\dhana\Downloads\Planetary_Gear_Detection\templates"
scale_percent_for_display = 60

# === LOAD VIDEO ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# === READ FIRST FRAME FOR TEMPLATE ===
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Cannot read first frame.")

gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
os.makedirs(template_folder, exist_ok=True)

# === SELECT TEMPLATES ===
rois = cv2.selectROIs("Select Templates", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Templates")

templates = []
for i, roi in enumerate(rois):
    x, y, w, h = roi
    if w > 0 and h > 0:
        template = gray_first[y:y+h, x:x+w]
        templates.append(template)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        template_path = os.path.join(template_folder, f"template_{i+1}_{timestamp}.png")
        cv2.imwrite(template_path, template)

# === CREATE WINDOWS AND TRACKBARS ===
cv2.namedWindow("Result")
cv2.createTrackbar("Threshold", "Result", 1, 20, nothing)
cv2.createTrackbar("Position", "Result", 0, total_frames - 1, nothing)

current_frame = 0
manual_seek = False

# === MAIN LOOP ===
while True:
    # Check if user moved the trackbar
    pos_from_bar = cv2.getTrackbarPos("Position", "Result")
    if pos_from_bar != current_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_from_bar)
        current_frame = pos_from_bar
        manual_seek = True
    else:
        manual_seek = False

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display_frame = frame.copy()
    threshold = cv2.getTrackbarPos("Threshold", "Result") / 20.0
    total_count = 0

    for template in templates:
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        boxes = []
        for pt in zip(*loc[::-1]):
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])

        nms_result = non_max_suppression_fast(boxes, 0.3)
        total_count += len(nms_result)

        for (x1, y1, x2, y2) in nms_result:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display count
    cv2.putText(display_frame, f'Count: {total_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resize for display
    display_resized = cv2.resize(display_frame, (
        int(display_frame.shape[1] * scale_percent_for_display / 100),
        int(display_frame.shape[0] * scale_percent_for_display / 100)
    ))

    # Update frame index trackbar
    cv2.setTrackbarPos("Position", "Result", int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    cv2.imshow("Result", display_resized)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

    current_frame += 1

cap.release()
cv2.destroyAllWindows()
