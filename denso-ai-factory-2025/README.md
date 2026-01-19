# DiffusionAD – Real-Time Industrial QC

DiffusionAD is a high-speed, high-accuracy anomaly detection system for industrial quality control using diffusion models. It detects defects without needing labeled defect images.

<p align="center">
  <img src="demo.png" alt="Demo Example" width="700">
</p>

---

## 🚀 Quick Start Guide

Follow these steps to set up and run the anomaly detection system.

### 1. Installation

**Prerequisites**: Python 3.8+

1.  **Clone the repository** (if not already done).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Running the Demo

The easiest way to use the system is via the interactive web interface.

```bash
      streamlit run demo/app.py
```

-   The app will open in your browser (usually at `http://localhost:8501`).
-   **Step 1**: In the Sidebar, **load a model**. You can select an existing model from `outputs/` or upload a `.pt` checkpoint.
-   **Step 2**: Select **"Single Image Analysis"**.
-   **Step 3**: Upload a test image.
-   **Step 4**: Click **"Detect Anomaly"**.

The results will be displayed in an interactive 2x2 grid (Input, Mask, Reconstruction, Heatmap) which supports zooming and panning.

### 3. Training a Model (Optional)

If you have your own dataset, you can train a new model.

1.  **Prepare Data**: Organize images in `datasets/<DatasetName>/train/good/`.
2.  **Configure**: Edit `args/args1.json` to match your dataset path and parameters.
3.  **Run Training**:
    ```bash
    python src/train.py
    ```
    Model checkpoints will be saved to `outputs/`.

---

## 📁 Project Structure

```
.
├── demo/
│   └── app.py            # Main Streamlit application
├── src/
│   ├── inference.py      # Core inference logic
│   ├── models.py         # Neural network architectures (UNet, Diffusion)
│   ├── train.py          # Training script
│   └── ...
├── args/
│   └── args1.json        # Configuration file
├── outputs/              # Saved models and logs
└── requirements.txt      # Python dependencies
```

## 🛠 Features

-   **Real-time Inference**: Optimized for low latency.
-   **Interactive Dashboard**: Zoom/Pan support for detailed defect inspection.
-   **Batch Processing**: Analyze multiple images at once.
-   **High Contrast Visualization**: clear distinction of anomalies.
