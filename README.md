
# ⚽ Tactical Football Path Predictor

This project predicts ideal player movement paths in football (soccer) matches based on player tracking data. It overlays both player trails and model-suggested trajectories onto a video and logs predictions for analysis.

## 📁 Project Structure

```
├── player_tracking_data.csv          # Input: Detected (x, y) positions per frame per player
├── your_clip.mp4                     # Input: Match video
├── path_dataset.csv                  # Auto-generated: Dataset used for model training
├── path_predictor_model.pth          # Output: Trained model
├── predicted_positions.csv           # Output: All model predictions with frame + errors
├── output_tactical_visualizer_*.avi  # Output: Video with overlays
├── train_path_predictor.py           # Train the path prediction model
├── create_path_dataset.py            # Build path_dataset.csv from tracking CSV
├── tactical_visualizer_model_predict.py         # Visualize model output on video
├── tactical_visualizer_bezier_export.py         # Adds Beziér trails + prediction export
├── positional_error_log.csv         # Optional: Positional prediction error logs
└── README.md                         # This file
```

---

## 🧠 What It Does

- Reads raw `(x, y)` tracking data for each player.
- Uses a trained neural network to **predict where each player should be going**.
- Overlays **actual paths**, **model suggestions**, and **errors** onto video.
- Exports a CSV for further analysis of model performance.

---

## 🛠 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- pandas
- scikit-learn

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run It

### 1️⃣ Prepare the tracking data
Make sure `player_tracking_data.csv` and your video `your_clip.mp4` are in the same folder.

### 2️⃣ Generate the dataset
```bash
python create_path_dataset.py
```

### 3️⃣ Train the model
```bash
python train_path_predictor.py
```

This will create `path_predictor_model.pth`.

### 4️⃣ Run the visualization with prediction CSV export
```bash
python tactical_visualizer_bezier_export.py
```

This creates:
- `output_tactical_visualizer_bezier_export.avi`
- `positional_error_log.csv` or `predicted_positions.csv`

---

## 📊 Epoch Log

If you want a log of training loss per epoch, the modified `train_path_predictor.py` now saves it to `epoch_losses.csv`.

---

## 💾 Export Instructions

To upload your repo to GitHub:

1. Initialize the repo:

```bash
git init
git add .
git commit -m "Initial commit: football path predictor"
```

2. Push to GitHub:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

---

## 🧠 Authors & Credits

Built by [You 💪] using PyTorch + OpenCV.
Inspired by tactical visualizations and player modeling in professional football.
