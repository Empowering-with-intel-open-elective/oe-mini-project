# OE Mini Project — Skin Lesion Cancer Detector

A full-stack web application that classifies a dermoscopic skin image as **Cancerous** or **Non-Cancerous** using a fine-tuned EfficientNet-B0 deep-learning model served through a Flask REST API and a React/Vite frontend.

---

## Table of Contents

1. [Problem Statement & Objectives](#problem-statement--objectives)
2. [Features & Workflow](#features--workflow)
3. [Technology Stack](#technology-stack)
4. [Architecture Overview](#architecture-overview)
5. [Prerequisites](#prerequisites)
6. [Installation & Setup](#installation--setup)
7. [Running the Application](#running-the-application)
8. [Using the Application](#using-the-application)
9. [Project Structure](#project-structure)
10. [Notebook](#notebook)
11. [Common Issues & Fixes](#common-issues--fixes)
12. [Known Limitations](#known-limitations)

---

## Problem Statement & Objectives

Skin cancer diagnosis from dermoscopic images traditionally requires expert dermatologists. This project aims to provide a lightweight tool that:

- Accepts an uploaded skin-lesion image.
- Runs inference with a pre-trained EfficientNet-B0 model trained on HAM10000-style classes.
- Returns a binary result — **Cancerous** or **Non-Cancerous** — via a clean web interface.

The seven HAM10000 lesion classes are mapped as follows:

| Class Index | HAM10000 Label | Binary Result |
|-------------|----------------|---------------|
| 0 | Akiec (Actinic keratoses) | Cancerous |
| 1 | Bcc (Basal cell carcinoma) | Cancerous |
| 2 | Bkl (Benign keratosis) | Non-Cancerous |
| 3 | Df (Dermatofibroma) | Non-Cancerous |
| 4 | Mel (Melanoma) | Cancerous |
| 5 | Nv (Melanocytic nevi) | Non-Cancerous |
| 6 | Vasc (Vascular lesions) | Non-Cancerous |

---

## Features & Workflow

1. User opens the React web app in a browser.
2. User uploads a skin image (PNG, JPG, etc.).
3. The frontend previews the image and sends it to the Flask backend via HTTP POST (`/predict`).
4. The backend preprocesses the image (resize → tensor → normalise) and runs it through EfficientNet-B0.
5. The predicted class index is mapped to **Cancerous** or **Non-Cancerous**.
6. The result is displayed on the frontend with visual styling.

---

## Technology Stack

### Backend

| Technology | Purpose |
|---|---|
| Python 3 | Runtime |
| Flask | REST API framework |
| Flask-CORS | Cross-origin request handling |
| PyTorch | Model inference engine |
| torchvision | EfficientNet-B0 architecture & transforms |
| Pillow (PIL) | Image loading |

### Frontend

| Technology | Purpose |
|---|---|
| React 19 | UI framework |
| Vite 8 | Dev server & bundler |
| Axios | HTTP client (calls `/predict`) |

---

## Architecture Overview

```
+-------------------------+    HTTP POST /predict     +------------------------------+
|  React Frontend         | ------------------------> |  Flask Backend               |
|  (Vite dev server)      |  multipart/form-data      |  http://127.0.0.1:5001       |
|  http://localhost:5173  |                           |                              |
|                         | <------------------------ |  EfficientNet-B0 (model.pth) |
|  Displays result        |  JSON: status, is_cancerous|  Binary classification      |
+-------------------------+                           +------------------------------+
```

- **Backend port:** `5001` (hardcoded in `app.py`)
- **Frontend port:** `5173` (Vite default; may auto-increment if the port is busy)
- The frontend calls `http://127.0.0.1:5001/predict` directly (hardcoded in `App.jsx`).

---

## Prerequisites

Make sure the following are installed and available on your `PATH`:

| Requirement | Minimum Version | Check command |
|---|---|---|
| Python | 3.8+ | `python --version` |
| pip | — | `pip --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |

> The model file `backend/model.pth` (~16 MB) must be present. It is already included in the repository.

---

## Installation & Setup

Open **two separate PowerShell terminals** — one for the backend and one for the frontend.

### Terminal 1 — Backend

```powershell
# 1. Navigate to the backend folder
#    NOTE: the folder name contains a leading space — quotes are required
cd "d:\Projects\oe-mini-project\ backend"

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install Python dependencies
pip install flask flask-cors torch torchvision pillow
```

> **Note:** If `torch` installation is slow or fails, visit https://pytorch.org/get-started/locally/ to get the correct install command for your system (CPU-only is sufficient for inference).

### Terminal 2 — Frontend

```powershell
# 1. Navigate to the frontend folder
cd "d:\Projects\oe-mini-project\frontend"

# 2. Install Node dependencies
npm install
```

---

## Running the Application

Both servers must be running simultaneously.

### Start the Backend (Terminal 1)

```powershell
# From inside the ' backend' directory, with the venv activated
python app.py
```

Expected output:

```
 * Running on http://127.0.0.1:5001
 * Debug mode: on
```

### Start the Frontend (Terminal 2)

```powershell
# From inside the 'frontend' directory
npm run dev
```

Expected output:

```
  VITE v8.x.x  ready in ...ms

  ➜  Local:   http://localhost:5173/
```

---

## Using the Application

1. Open your browser and go to **http://localhost:5173**.
2. Click the upload area (`+`) and select a skin lesion image.
   - Five sample PNG images are available in the `test_samples/` folder.
3. Click **Analysis**.
4. The result — **Cancerous** or **Non-Cancerous** — is displayed below the image.

---

## Project Structure

```
oe-mini-project/
├──  backend/               # Flask API (note the leading space in the folder name)
│   ├── app.py              # API entry point — defines the /predict endpoint
│   └── model.pth           # Pre-trained EfficientNet-B0 weights (~16 MB)
├── frontend/               # React/Vite web app
│   ├── src/
│   │   ├── App.jsx         # Main component — upload, preview, API call, result
│   │   ├── index.css       # Global styles
│   │   └── main.jsx        # React entry point
│   ├── package.json        # Node dependencies and npm scripts
│   └── vite.config.js      # Vite configuration
├── test_samples/           # 5 sample skin lesion PNG images for testing
├── OE_Fixed.ipynb          # Jupyter notebook (model training / exploration)
├── interview_prep.md       # Project notes
└── README.md               # This file
```

---

## Notebook

`OE_Fixed.ipynb` contains the model training and/or evaluation workflow. To open it:

```powershell
# Install Jupyter if not already installed (with venv active)
pip install jupyter

# Launch from the project root
jupyter notebook OE_Fixed.ipynb
```

The notebook is independent of the web application — you do not need to run it to use the app.

---

## Common Issues & Fixes

| Issue | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: flask` | Dependencies not installed | Run `pip install flask flask-cors torch torchvision pillow` with the venv active |
| `FileNotFoundError: model.pth` | Running `app.py` from the wrong directory | `cd` into the ` backend/` folder before running `python app.py` |
| Browser alert: "Error connecting to the backend" | Flask server not running or wrong port | Ensure `python app.py` is running and listening on port `5001` |
| Frontend shows a blank page | Node modules missing | Run `npm install` inside `frontend/` |
| Port `5173` already in use | Another Vite instance running | Vite auto-selects the next available port; check the terminal output for the actual URL |
| PowerShell blocks `Activate.ps1` | Execution policy restriction | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry |

---

## Known Limitations

- The backend URL (`http://127.0.0.1:5001`) is hardcoded in `App.jsx`. Changing the backend port requires updating that file.
- The backend folder name has a **leading space** (` backend`), which can cause issues with tab completion. Always wrap the path in quotes.
- The model runs on **CPU** by default (`map_location="cpu"`). Inference may be slow on older machines.
- The `/predict` endpoint accepts only image files that Pillow can open. Non-image files will return a `500` error.
- No confidence score is currently surfaced in the UI — only the binary classification label is shown.
