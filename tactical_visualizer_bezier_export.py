
import cv2
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

# ==== CONFIG ====
VIDEO_PATH = "your_clip.mp4"
PLAYER_DATA_PATH = "player_tracking_data.csv"
SAVE_PATH = "output_tactical_visualizer_bezier_export.avi"
CSV_OUTPUT = "predicted_positions.csv"
NUM_FRAMES = 150
TRAIL_LENGTH = 20
MODEL_PATH = "path_predictor_model.pth"

# ==== MODEL ====
classifier = nn.Sequential(
    nn.Linear(10, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
classifier.load_state_dict(torch.load(MODEL_PATH))
classifier.eval()

# ==== TRACKING ====
player_trails = defaultdict(list)
ideal_trails = defaultdict(list)
player_data = pd.read_csv(PLAYER_DATA_PATH)

# Collect predictions for export
predicted_rows = []

# ==== VIDEO SETUP ====
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc(*'XVID'), fps, (FRAME_WIDTH, FRAME_HEIGHT))

frame_count = 0

def bezier_curve(points, num=50):
    points = np.array(points)
    n = len(points) - 1
    result = []
    for t in np.linspace(0, 1, num):
        p = np.zeros(2)
        for i in range(n + 1):
            binom = np.math.comb(n, i)
            p += binom * ((1 - t)**(n - i)) * (t**i) * points[i]
        result.append(tuple(p))
    return result

while cap.isOpened() and frame_count < NUM_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    frame_players = player_data[player_data['frame'] == frame_count]
    for _, row in frame_players.iterrows():
        pid = int(row['track_id'])
        x, y = int(row['x_center']), int(row['y_center'])

        player_trails[pid].append((x, y))
        if len(player_trails[pid]) > TRAIL_LENGTH:
            player_trails[pid].pop(0)

        if len(player_trails[pid]) < 5:
            continue

        history = player_trails[pid][-5:]
        flattened = [v for point in history for v in point]
        input_tensor = torch.tensor(flattened, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = classifier(input_tensor)
        target_x, target_y = int(output[0][0].item()), int(output[0][1].item())

        predicted_rows.append({
            "frame": frame_count,
            "player_id": pid,
            "current_x": x,
            "current_y": y,
            "predicted_x": target_x,
            "predicted_y": target_y
        })

        ideal_trails[pid].append((target_x, target_y))
        if len(ideal_trails[pid]) > TRAIL_LENGTH:
            ideal_trails[pid].pop(0)

        # Draw player and prediction
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        cv2.circle(frame, (target_x, target_y), 5, (255, 0, 0), -1)
        cv2.line(frame, (x, y), (target_x, target_y), (0, 255, 0), 2)

        # Draw curved trails
        if len(ideal_trails[pid]) >= 3:
            curve_points = bezier_curve(ideal_trails[pid])
            for i in range(1, len(curve_points)):
                cv2.line(frame, (int(curve_points[i - 1][0]), int(curve_points[i - 1][1])),
                         (int(curve_points[i][0]), int(curve_points[i][1])), (128, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Visualizer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Export CSV
pd.DataFrame(predicted_rows).to_csv(CSV_OUTPUT, index=False)
print("✅ Video saved to:", SAVE_PATH)
print("📄 CSV saved to:", CSV_OUTPUT)
