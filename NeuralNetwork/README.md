# Project 3 — CNN: Cats vs. Dogs

By: Preston Brownlee

AI Assistance: Chat-GPT

This project trains and evaluates a Convolutional Neural Network (CNN) to classify images of cats and dogs.  
Everything is implemented and documented in the Jupyter notebook: report.ipynb.

---

## Folder Layout

Project3/

├─ report.ipynb # Main notebook 

├─ data/

│ ├─ train/

│ │ ├─ cats/.jpg

│ │ └─ dogs/.jpg

│ └─ test/

│ ├─ cats/.jpg

│ └─ dogs/.jpg

├─ class_names.txt

└─ history.json


If data/ is missing, the notebook downloads it from Kaggle automatically (see below).

---

## Quick Start (Windows, PowerShell)

From the repo root:

```powershell
# 1) Create & activate Python 3.11 venv
py -3.11 -m venv .venv311
.venv311\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install --upgrade pip
pip install tensorflow==2.20.* numpy matplotlib scikit-learn kaggle

```

## If Data Is Missing

Kaggle Credentials (One-Time)

On Kaggle, go to Account → Create New API Token (downloads kaggle.json).

Place kaggle.json here:

%USERPROFILE%\.kaggle\kaggle.json


Make sure it’s readable by your user account.

The notebook uses the Kaggle API to download the dataset:

Dataset: Cats and Dogs Image Classification (Samuel Cortinhas)
https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification/data



## Reproduce Training & Evaluation

Open Project3/report.ipynb in VS Code and select the kernel:

...\Streamlit-Multi-App\.venv311\Scripts\python.exe


Then run the notebook top-to-bottom. Key steps:

Step 1–10: dataset selection & description, imports, CNN definition
(Conv → ReLU → MaxPool ×3 → Flatten → Dense(128, ReLU) → Dense(softmax)).

Step 11A–11E: build datasets, train with callbacks
(EarlyStopping, ReduceLROnPlateau, ModelCheckpoint(save_best_only=True)),
reload the best checkpoint, evaluate on test, plot accuracy/loss, and save artifacts:


class_names.txt

history.json

Expected Outputs

Accuracy/Loss plots (train vs val)

Final test accuracy & loss

Classification report and confusion matrix

Typical run:
Best val accuracy ≈ 0.90, test accuracy ≈ 0.66–0.71.