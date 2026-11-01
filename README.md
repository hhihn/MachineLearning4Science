---
title: ML Science Lab
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
---

# ML Science Lab — Instructor starter kit

**Audience:** scientists with little/no Python experience (up to 20 people).  
**Architecture:** a public Hugging Face Space with this Gradio app, plus three Google Colab notebooks distributed via a GitHub repository.

## Contents

- `app.py`: fast interactive scientific tabular ML explorer, reports validation scores only.
- `automl_app.py`: standalone AutoML app with CSV upload, cross-validated model tuning, held-out evaluation, pipeline export, and prediction on new data.
- `styles.css`: shared minimal blue stylesheet for both apps.
- `notebooks/01_tabular_basics.ipynb`: interpretable tabular ML with scikit-learn and chemistry data.
- `notebooks/02_automl_classification.ipynb`: reproducible FLAML AutoML classification, CV, held-out evaluation.
- `notebooks/03_automl_regression.ipynb`: diabetes regression and cost-aware AutoML.
- `WORKSHOP_PLAN.md`: suggested schedule, learning goals, exercise prompts.
- `FACILITATOR_CHECKLIST.md`: preflight and contingencies.

## Deploy the web application

1. Create or log in to an HF account eligible to create Gradio Spaces (the current plan requirement is PRO or an eligible organizational plan).
2. Open https://huggingface.co/new-space and select **Gradio SDK**, **public**, and **CPU Upgrade** (8 vCPU, 32 GB RAM, advertised $0.03/hour).
3. Upload `README.md`, `app.py`, `styles.css`, and `requirements.txt` from this folder **into the root** of the Space repository. You can also push through Git. To serve the separate AutoML app instead, also upload `automl_app.py` and change `app_file` in the README metadata to `automl_app.py`.
4. Allow the Space build to complete; open the public Space URL and test both datasets/models.
5. Leave the Space public only if datasets and code are safe to share. Never publish participant data or credentials.
6. When the workshop is over, switch to the free/basic instance to avoid ongoing upgrade costs. The subscription itself may remain billable until cancelled.

**Note:** the Space hardware details and plan prerequisites change; confirm current pricing in HF settings before enabling upgrades.

## Publish the guided notebooks

1. Create a public GitHub repository, such as `ml-science-workshop`.
2. Add the entire starter-kit folder's contents, including `notebooks`.
3. Construct a Colab link for each notebook:
   `https://colab.research.google.com/github/OWNER/REPO/blob/main/notebooks/01_tabular_basics.ipynb`
   (change `OWNER` and `REPO` to your actual GitHub path).
4. Verify all notebook links in an incognito window. Participants need access to a Google account to run/save their own notebooks. In Colab, select **File → Save a copy in Drive** before making edits.
5. Use default **CPU runtime**. The AutoML notebooks install `flaml[automl]` in the first cell, via pip.
6. Give participants only the Space URL and three notebook URLs in a short landing page or workshop slide.

## Run the app locally (instructor only)

```bash
python -m pip install -r requirements.txt
python app.py
```

Run the separate AutoML app with the same environment:

```bash
python automl_app.py
```

Upload a CSV (automatically selected as the active dataset), choose its separator,
target, and classification or regression task, and optionally exclude ID or leakage
columns. No CSV preview is displayed. Built-in Wine and Diabetes data also work.
The app reserves 20% of rows for testing, searches seven model families plus a dummy
baseline using 3–5 development CV folds, and selects the best learned model by macro
F1 or RMSE. Imputation, scaling, and categorical encoding are learned inside each
fold. The AutoML budget sets 7–56 total hyperparameter configurations across the
seven model families (default 21); the dummy baseline adds one configuration.
Each configuration requires one fit per CV fold, followed by eight refits of the
best configurations including the baseline. The UI shows this training-work estimate.
This is a search-work budget rather than a time or monetary limit. Increase it to
explore more hyperparameters; remaining trials are spread evenly across models
with available configurations. The report records each model's allocation.

Download the fitted pipeline, metrics, leaderboard, fold scores, test predictions,
feature importance, and 300 DPI PNG/vector PDF figures in one ZIP. Tables display
three decimal places; exported CSVs retain full precision. Use the prediction section
to apply the selected pipeline to another CSV with the same input column names.
The pipeline is fitted on the development partition; test data are excluded from fitting.
These random splits assume independent rows; longitudinal or grouped datasets need
a suitable evaluation strategy. Repeated test inspection can bias model selection.
AutoML runs are queued one at a time and may take longer than the original explorer.

## Technical strategy for 20 participants

- Gradio intentionally trains **fast scikit-learn models**. No FLAML search runs on the shared server.
- Each participant performs the time-budgeted FLAML searches on their **own Colab CPU runtime**.
- App requests are queued with concurrency limit 4, and each estimator is restricted to one CPU worker. This is a starting configuration, not a concurrency guarantee.
- For methodological correctness, app model changes use **validation** scores; a reserved test set is not revealed during exploration. Notebooks demonstrate final test evaluation.
- Keep datasets public and tiny. Do not upload identifiable scientific or patient data.

## Data and sources

- The built-in scikit-learn Wine chemistry and Diabetes regression datasets are used for the working app. See https://scikit-learn.org/stable/datasets/toy_dataset.html
- Additional materials-science exercise: UCI Concrete Compressive Strength, 1,030 observations, 8 numeric features, regression, dataset DOI https://doi.org/10.24432/C5PK67 (CC BY 4.0). Download and bundle the public CSV in advance if your workshop network could be unreliable.
- FLAML: https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/

This repo is an educational demonstration, not medical guidance. The `license: mit` above applies to the original workshop code, not to third-party datasets.

## Kumo Science Lab (inference only)

Run `pip install -r requirements-kumo.txt` and `python kumo_app.py` with
Python 3.12. This app offers small, medium and large NVIDIA Kumo classification
checkpoints, Wine chemistry, labeled CSV evaluation, and new CSV predictions.
No training or fine-tuning occurs. Four inference ensemble members are used.

For a separate dedicated GPU Gradio Space, upload `kumo_app.py`, `styles.css`,
and copy `requirements-kumo.txt` as the Space's root `requirements.txt`.
Use this README metadata in that Space:

```yaml
---
title: Kumo Science Lab
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.5.1
python_version: '3.12'
app_file: kumo_app.py
models:
  - nvidia/Kumo-Tabular
license: mit
---
```

Select dedicated CUDA hardware in Space settings. CPU fallback supports local
checks but can be slow. Downloads occur on first use of each checkpoint; only
one size stays resident. Requests are serialized. Uploaded context remains in
browser-session state on the server; upload only suitable public teaching data.
The original app remains the default entrypoint in this workspace.

## SAM Segmentation Science Lab

Use Python 3.12 and install `pip install -r requirements-sam.txt`, then run
`python sam_app.py`. Upload a PNG/JPEG or select the webcam source and capture
a snapshot, then click foreground/background points,
then select **Segment object**, or select **Find masks** for automatic proposals.
Images are processed at a maximum 1,024-pixel side; downloads use that resolution.
The Color overlay tab highlights masks on the source image at 85% opacity,
with distinct colors and smaller masks on top in overlapping regions. The
Individual masks tab preserves the binary previews.
The ZIP includes individual binary PNG masks, the same color overlay, and JSON scores,
areas, prompts, settings and timing. Scores estimate quality, not measured accuracy.
SAM proposes potentially overlapping regions without class names.

Webcam capture requires browser camera permission and localhost or HTTPS
(such as a Hugging Face Space). Capture a still image before segmenting;
this app does not continuously process video. Taking a new snapshot or
removing the image resets points, results and downloads.

Only `facebook/sam-vit-base` is loaded, on first inference. Download access is
required initially; cached weights can be reused. CUDA is selected when available,
otherwise CPU. Requests are serialized. Higher sampling densities cost more time
and memory. No training occurs.

For a separate GPU Gradio Space, upload `sam_app.py`, `styles.css`, this README,
and `requirements-sam.txt` renamed to `requirements.txt`. Set README metadata
`app_file: sam_app.py`, `python_version: '3.12'`, and
`models: [facebook/sam-vit-base]`, then select dedicated CUDA hardware.
The original workshop app remains the default entrypoint here.

## Multilingual Zero-Shot Text Science Lab

With Python 3.12, run `pip install -r requirements-zeroshot.txt`, then
`python zeroshot_app.py`. Paste one text and 2–20 labels, one per line.
Single-label mode chooses the highest score; multi-label mode selects independent
scores at or above the threshold (default 0.5). Adjust the hypothesis template
for your task or language. Inputs exceeding 512 tokens including the hypothesis
are rejected rather than truncated. JSON exports preserve all scores and settings.

The fixed MIT-licensed `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` checkpoint runs
in float32 on CUDA when available, otherwise CPU. Weights download on first use;
requests are serialized. No training or hosted inference API is used. Scores
are not calibrated accuracy estimates; performance varies by language.

For a separate Gradio Space, upload `zeroshot_app.py`, `styles.css`, and this
README; copy `requirements-zeroshot.txt` as root `requirements.txt`. Set README
metadata `app_file: zeroshot_app.py`, `python_version: '3.12'`, and
`models: [MoritzLaurer/mDeBERTa-v3-base-mnli-xnli]`. CPU and dedicated CUDA hardware
are supported. The original default app remains unchanged.
