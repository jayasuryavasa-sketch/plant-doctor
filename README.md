# Plant Doctor — AI Plant Disease Detection & Care System

Plant Doctor is a responsive Flask web app for AI-assisted leaf-image checks. It provides image upload, browser camera capture, validation, clear uncertainty handling, local scan history, a disease guide, and responsible general plant-care information.

> **Important:** Results are informational—not a guaranteed diagnosis or cure. For severe, unusual, or uncertain cases, consult a qualified agricultural professional or local agricultural extension service.

## Features

- Botanical Green + Soft Cream responsive HTML/CSS/JavaScript interface
- Upload JPG/JPEG/PNG/WEBP and capture a photo with a supported browser camera
- File/type/size/content, minimum-dimension, and very-dark-image checks
- Modular predictor with a configurable confidence threshold
- Clearly marked **demo mode** until real model weights are trained and added
- Separate JSON disease-information layer (not embedded in model code)
- SQLite scan history with result viewing and deletion
- Searchable/filterable Disease Guide, About, and responsible-use content
- CPU-friendly MobileNetV3-small transfer-learning trainer and evaluator

## Quick start on Windows

Open **Command Prompt** in this `plant-doctor` folder and run:

```bat
py -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser. Stop the server with `Ctrl+C`. On a later day, activate the environment again with `venv\Scripts\activate` before running `python app.py`.

If `py` is not available, replace it with `python`. Verify either one with `py --version` or `python --version`. Python 3.10–3.12 is a good choice.

## Demo mode vs. real model

The app runs immediately without downloading a large ML package or fabricating accuracy. In **demo mode**, predictions deliberately stay below the confidence threshold and the user sees an uncertainty message. This lets you demonstrate all UI, validation, database, guide, camera, and error-handling features honestly.

To enable real inference, install the ML packages and put a trained checkpoint in `model/plant_disease_model.pth`:

```bat
pip install -r requirements-ml.txt
python training\train_model.py --data dataset --epochs 8 --batch-size 8
python app.py
```

Restart the server after training. The page will then show **Trained** model mode. The trainer writes `model/class_names.json` and the best validation checkpoint. Do not report its validation score as field accuracy.

### Optional hosted model API (no local training)

You can also use a free-tier Hugging Face hosted model. This is convenient for a demonstration but has rate/availability limits and sends the image to that service. It is **not** guaranteed to be the highest-accuracy model for your plants or local conditions. The included default is a public MobileNetV3 model trained on PlantVillage classes; its model card reports its own test result, which should not be treated as a field result. [Model card](https://huggingface.co/imaflower/plantvillage-mobilenetv3) · [Hugging Face image-classification API docs](https://huggingface.co/docs/inference-providers/en/tasks/image-classification)

1. Create a Hugging Face account and make a personal access token with **Inference Providers** permission.
2. In the project folder, copy `.env.example` to `.env`.
3. Open `.env`, paste your own token after `HF_TOKEN=`, and save it. Never share or commit this file.
4. Restart `python app.py`.

When successful, a result shows **Hosted API** mode. If the token is missing, invalid, rate-limited, or the API is unavailable, the app safely returns to the marked demo/uncertain mode.

## Dataset preparation

Obtain a PlantVillage-compatible dataset only under terms that allow your use. Create this layout, keeping class folder names consistent across all three folders:

```text
dataset/
  train/
    Tomato___Early_blight/
  validation/
    Tomato___Early_blight/
  test/
    Tomato___Early_blight/
```

Use about 70–80% training, 10–15% validation, and 10–15% testing images, split by image and ideally by source/plant where possible. Training learns patterns; validation selects the best checkpoint; testing is held back for final reporting. The script uses sensible, light augmentation (crop, horizontal flip, and small rotation), 224px inputs, batch size 8, no worker processes, early stopping, and CPU compatibility. Reduce `--batch-size` to 4 if memory is limited.

Evaluate only after training:

```bat
python training\evaluate_model.py --data dataset\test
```

It prints accuracy-related classification metrics (precision, recall, F1) and a confusion matrix. Real-world performance can be lower because of lighting, backgrounds, unsupported plants, pests, mixed conditions, and dataset bias.

## Configuration and architecture

Set `CONFIDENCE_THRESHOLD` (default `0.65`) before starting if needed:

```bat
set CONFIDENCE_THRESHOLD=0.70
python app.py
```

`app.py` owns routes and SQLite; `utils/image_validation.py` protects uploads; `utils/prediction.py` owns inference; `data/diseases.json` owns readable guidance; templates and static files own the frontend. Uploads and the local database are intentionally excluded from Git.

## Testing

Install the base dependencies, then run:

```bat
python -m pytest
```

Manual checks: invalid extension, oversized file, dark image, clear upload, camera permission denied, browser mobile view, history deletion, demo low-confidence result, and real-model fallback if the checkpoint is missing.

## Deployment (Render)

1. Create a GitHub repository, then from this folder run:

   ```bat
   git init
   git add .
   git commit -m "Initial Plant Doctor project"
   git branch -M main
   git remote add origin YOUR_GITHUB_REPOSITORY_URL
   git push -u origin main
   ```

   `git init` starts version control; `git add .` stages project files; `git commit` makes a snapshot; `branch -M main` names the primary branch; `remote add` connects GitHub; `push` uploads it.

2. In Render, create a **Web Service** from that repository. Set build command to `pip install -r requirements.txt` and start command to `gunicorn app:app`.
3. Add a long random `SECRET_KEY` environment variable. If using hosted inference, add `HF_TOKEN` and `HF_MODEL` as Render environment variables—never commit `.env`. For real local-model inference, use `requirements-ml.txt` instead and arrange licensed model storage/building. Do not commit large weights or private data.
4. Render's local disk is ephemeral: uploads and SQLite history may reset after deploy. Use managed storage/database for a production system.

## Project documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for an abstract, problem statement, modules, requirements, future scope, conclusion, demo flow, and viva answers.

## Technology

Python, Flask, SQLite, Pillow, HTML, CSS, JavaScript; optional PyTorch, torchvision, and scikit-learn.
