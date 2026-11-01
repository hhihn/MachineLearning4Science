"""Transfer Learning Science Lab: python transfer_app.py.

Frozen ImageNet features + a new trainable leaf classifier (linear probing).
Only official training and validation data are loaded; test data stay reserved.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter
import json
import tempfile
import zipfile

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

MODEL_ID = "google/mobilenet_v2_1.0_224"
MODEL_REVISION = "75e607b"
DATASET_ID = "AI-Lab-Makerere/beans"
DATASET_REVISION = "27aa014"
CLASS_NAMES = ["Angular leaf spot", "Bean rust", "Healthy"]
TRANSFER = "Pretrained features + new head"
PIXELS = "Resized pixels + new head"
LOCK = Lock()


def validate_settings(per_class, epochs, learning_rate, seed):
    values = (per_class, epochs, learning_rate, seed)
    try:
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError()
        if any(float(value) != int(value) for value in (per_class, epochs, seed)):
            raise ValueError()
        per_class, epochs, seed = int(per_class), int(epochs), int(seed)
        learning_rate = float(learning_rate)
        if not (5 <= per_class <= 100 and 1 <= epochs <= 100
                and 0.0001 <= learning_rate <= 0.1 and 0 <= seed <= 10000):
            raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise gr.Error("Use 5–100 images per class, 1–100 epochs, a learning rate "
                       "of 0.0001–0.1, and an integer seed from 0 to 10,000.") from None
    return per_class, epochs, learning_rate, seed


def balanced_indices(labels, per_class, seed):
    """Nested balanced subsets: increasing the budget retains earlier samples."""
    rng = np.random.default_rng(seed)
    selected = []
    labels = np.asarray(labels)
    for label in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(f"Class {label} has fewer than {per_class} images.")
        selected.extend(rng.permutation(candidates)[:per_class].tolist())
    return selected


@lru_cache(maxsize=1)
def load_data():
    from datasets import load_dataset

    # Use the Parquet builder with explicit files: the Hub's default config
    # requires all three splits, including the test split we want to reserve.
    prefix = f"hf://datasets/{DATASET_ID}@{DATASET_REVISION}/data"
    data = load_dataset(
        "parquet",
        data_files={"train": f"{prefix}/train-00000-of-00001.parquet",
                    "validation": f"{prefix}/validation-00000-of-00001.parquet"},
    )
    expected = ["angular_leaf_spot", "bean_rust", "healthy"]
    for split in ("train", "validation"):
        if data[split].features["labels"].names != expected:
            raise ValueError("Dataset label mapping differs from the lesson's expected classes.")
    return data


@lru_cache(maxsize=1)
def load_backbone():
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=False,
    )
    source = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_safetensors=True,
    )
    backbone = source.base_model.to(device).eval()
    backbone.requires_grad_(False)
    return processor, backbone, device


def extract_features(dataset, indices, processor, backbone, device, progress):
    import torch

    embeddings, pixels, labels = [], [], []
    # Keep image decoding and model activations bounded even on CPU.
    for offset in range(0, len(indices), 8):
        batch = [dataset[int(index)] for index in indices[offset:offset + 8]]
        images = [row["image"].convert("RGB") for row in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            features = backbone(**inputs).pooler_output
        embeddings.append(features.cpu().numpy().copy())
        pixels.extend(np.asarray(image.resize((16, 16), Image.Resampling.BILINEAR),
                                 dtype=np.float32).reshape(-1) / 255 for image in images)
        labels.extend(int(row["labels"]) for row in batch)
        progress(min((offset + len(batch)) / len(indices), 1), desc="Extracting frozen image features")
    return np.concatenate(embeddings), np.asarray(pixels), np.asarray(labels, dtype=np.int64)


@lru_cache(maxsize=2)
def cached_features(per_class, seed):
    """Cache only immutable representations; classifiers belong to each run."""
    data = load_data()
    indices = balanced_indices(data["train"]["labels"], per_class, seed)
    processor, backbone, device = load_backbone()
    progress = gr.Progress()
    train = extract_features(data["train"], indices, processor, backbone, device, progress)
    validation = validation_features()
    return train, validation, indices


@lru_cache(maxsize=1)
def validation_features():
    data = load_data()["validation"]
    processor, backbone, device = load_backbone()
    return extract_features(data, list(range(len(data))), processor, backbone, device, gr.Progress())


def fit_head(train_features, train_labels, val_features, val_labels,
             epochs, learning_rate, seed, approach, progress):
    """Train a linear softmax head; feature scaling learns from training only."""
    import torch

    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x_train = torch.from_numpy(((train_features - mean) / scale).astype(np.float32))
    x_val = torch.from_numpy(((val_features - mean) / scale).astype(np.float32))
    y_train = torch.as_tensor(train_labels, dtype=torch.long)
    y_val = torch.as_tensor(val_labels, dtype=torch.long)
    # Local seeded initialization does not change other sessions' RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = torch.nn.Linear(x_train.shape[1], len(CLASS_NAMES))
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []
    initial_predictions = None
    for epoch in range(epochs + 1):
        if epoch:
            head.train()
            optimizer.zero_grad()
            loss = loss_fn(head(x_train), y_train)
            if not torch.isfinite(loss):
                raise ValueError("Training diverged. Try a smaller learning rate.")
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            for split, features, labels in (("Train", x_train, y_train),
                                            ("Validation", x_val, y_val)):
                logits = head(features)
                short_name = "Transfer" if approach == TRANSFER else "Pixels"
                history.append({"Epoch": epoch, "Approach": approach, "Split": split,
                                "Series": f"{short_name} · {split.lower()}",
                                "Loss": float(loss_fn(logits, labels)),
                                "Accuracy": float((logits.argmax(1) == labels).float().mean())})
            probabilities = head(x_val).softmax(dim=1).numpy()
        if epoch == 0:
            initial_predictions = probabilities.argmax(axis=1).copy()
        progress(epoch / epochs, desc=f"Training {approach.lower()} · epoch {epoch}/{epochs}")
    fitted = {"weight": head.weight.detach().numpy().copy(),
              "bias": head.bias.detach().numpy().copy(), "mean": mean, "scale": scale}
    return fitted, history, probabilities, initial_predictions


def head_probabilities(features, fitted):
    features = (features - fitted["mean"]) / fitted["scale"]
    logits = features @ fitted["weight"].T + fitted["bias"]
    logits -= logits.max(axis=1, keepdims=True)
    scores = np.exp(logits)
    return scores / scores.sum(axis=1, keepdims=True)


def metric_row(name, train_true, train_pred, val_true, val_pred, parameters):
    return {"Approach": name, "Train accuracy": accuracy_score(train_true, train_pred),
            "Validation accuracy": accuracy_score(val_true, val_pred),
            "Validation macro F1": f1_score(val_true, val_pred, labels=[0, 1, 2],
                                             average="macro", zero_division=0),
            "Trainable parameters": parameters}


def make_archive(report, curves, predictions, fitted):
    directory = Path(tempfile.mkdtemp(prefix="transfer_results_"))
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    curves.to_csv(directory / "learning_curves.csv", index=False)
    predictions.to_csv(directory / "validation_predictions.csv", index=False)
    np.savez(directory / "leaf_head.npz", **fitted)
    (directory / "README.txt").write_text(
        "leaf_head.npz contains weight, bias, mean and scale for the transferred head.\n"
        "Use the pinned model and image processor in report.json to extract pooler_output.\n"
        "Compute z = (features - mean) / scale, then softmax(z @ weight.T + bias).\n"
        "Output columns follow class_names in report.json.\n"
        "Metrics use validation data for exploration, not the reserved test split.\n", encoding="utf-8",
    )
    archive = directory / "transfer_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for filename in ("report.json", "learning_curves.csv", "validation_predictions.csv",
                         "leaf_head.npz", "README.txt"):
            bundle.write(directory / filename, filename)
    return str(archive)


def run_experiment(per_class, epochs, learning_rate, seed, progress=gr.Progress()):
    per_class, epochs, learning_rate, seed = validate_settings(per_class, epochs, learning_rate, seed)
    start = perf_counter()
    try:
        with LOCK:
            progress(0, desc="Loading model and leaf images; first use downloads about 177 MB")
            (train_x, train_pixels, train_y), (val_x, val_pixels, val_y), indices = cached_features(per_class, seed)
            prepared_seconds = perf_counter() - start
            head_start = perf_counter()
            fitted, history, probabilities, before = fit_head(
                train_x, train_y, val_x, val_y, epochs, learning_rate, seed, TRANSFER, progress,
            )
            pixel_head, pixel_history, pixel_probabilities, _ = fit_head(
                train_pixels, train_y, val_pixels, val_y, epochs, learning_rate, seed, PIXELS, progress,
            )
            head_seconds = perf_counter() - head_start
            predicted = probabilities.argmax(axis=1)
            rows = [metric_row(TRANSFER, train_y, head_probabilities(train_x, fitted).argmax(1),
                               val_y, predicted, fitted["weight"].size + fitted["bias"].size),
                    metric_row(PIXELS, train_y, head_probabilities(train_pixels, pixel_head).argmax(1),
                               val_y, pixel_probabilities.argmax(1),
                               pixel_head["weight"].size + pixel_head["bias"].size)]
            comparison = pd.DataFrame(rows)
            curves = pd.DataFrame(history + pixel_history)
            confusion = pd.DataFrame(confusion_matrix(val_y, predicted, labels=[0, 1, 2]),
                                     columns=[f"Predicted: {name}" for name in CLASS_NAMES])
            confusion.insert(0, "Actual class", CLASS_NAMES)
            predictions = pd.DataFrame({"Validation index": np.arange(len(val_y)),
                                        "Actual": [CLASS_NAMES[i] for i in val_y],
                                        "Predicted": [CLASS_NAMES[i] for i in predicted],
                                        "Score": probabilities.max(axis=1),
                                        "Correct": predicted == val_y})
            data = load_data()
            examples = []
            for label in range(3):
                for position in np.flatnonzero(train_y == label)[:2]:
                    image = data["train"][indices[int(position)]]["image"].convert("RGB")
                    image.thumbnail((240, 240))
                    examples.append((image, f"Train · {CLASS_NAMES[label]}"))
            order = np.concatenate([np.flatnonzero(predicted != val_y),
                                    np.flatnonzero(predicted == val_y)])[:9]
            errors = []
            for index in order:
                image = data["validation"][int(index)]["image"].convert("RGB")
                image.thumbnail((240, 240))
                errors.append((image, f"#{index} · Actual: {CLASS_NAMES[val_y[index]]} · "
                               f"Predicted: {CLASS_NAMES[predicted[index]]} ({probabilities[index].max():.2f})"))
            _, backbone, device = load_backbone()
            report = {"model": MODEL_ID, "model_revision": MODEL_REVISION,
                      "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
                      "class_names": CLASS_NAMES, "method": "Frozen backbone; train a linear softmax head",
                      "settings": {"images_per_class": per_class, "epochs": epochs,
                                   "learning_rate": learning_rate, "seed": seed,
                                   "optimizer": "Adam", "weight_decay": 0.001},
                      "training_indices": indices, "training_size": len(train_y),
                      "validation_size": len(val_y), "test_evaluated": False,
                      "feature_dimensions": int(train_x.shape[1]),
                      "frozen_backbone_parameters": sum(p.numel() for p in backbone.parameters()),
                      "device": device, "preparation_seconds": prepared_seconds,
                      "head_training_seconds": head_seconds,
                      "validation_accuracy_before_training": accuracy_score(val_y, before),
                      "metrics": rows,
                      "dataset_fingerprints": {key: data[key]._fingerprint for key in ("train", "validation")}}
            archive = make_archive(report, curves, predictions, fitted)
            state = {"head": fitted, "settings": report["settings"]}
        accuracy = rows[0]["Validation accuracy"]
        status = (f"**Validation accuracy: {accuracy:.1%}** · resized-pixel baseline: "
                  f"{rows[1]['Validation accuracy']:.1%}.\n\n"
                  f"{len(train_y)} training images ({per_class}/class) → frozen MobileNetV2 → "
                  f"{train_x.shape[1]:,} features → **{rows[0]['Trainable parameters']:,} trainable parameters**. "
                  f"{report['frozen_backbone_parameters']:,} backbone parameters stay frozen.\n\n"
                  f"All {len(val_y)} validation images · random head before training: "
                  f"{report['validation_accuracy_before_training']:.1%} · {device} · "
                  f"load/features {prepared_seconds:.1f}s · both heads {head_seconds:.1f}s. "
                  "Epoch 0 in the curves shows the untrained heads. Test data remain reserved.")
        display_metrics = comparison.replace({"Approach": {TRANSFER: "Transfer", PIXELS: "Pixels"}}).rename(
            columns={"Train accuracy": "Train acc.", "Validation accuracy": "Validation acc.",
                     "Validation macro F1": "Macro F1", "Trainable parameters": "Trainable params"},
        ).round(3)
        return state, status, display_metrics, curves, curves, confusion, predictions, examples, errors, archive, None
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error("Experiment failed. Install requirements-transfer.txt and check Hugging Face "
                       f"download access, disk space and memory. Details: {exc}") from exc


def predict_image(image, state):
    if state is None:
        raise gr.Error("Train a transfer learning head first using the current settings.")
    if image is None:
        raise gr.Error("Upload a leaf image first.")
    try:
        import torch

        image = ImageOps.exif_transpose(image).convert("RGB")
        with LOCK:
            processor, backbone, device = load_backbone()
            with torch.no_grad():
                features = backbone(**processor(images=image, return_tensors="pt").to(device)).pooler_output
            probabilities = head_probabilities(features.cpu().numpy(), state["head"])[0]
        return {name: float(value) for name, value in zip(CLASS_NAMES, probabilities)}
    except Exception as exc:
        raise gr.Error(f"Could not classify this image: {exc}") from exc


def clear_results():
    return (None, "Settings changed. Train again to update results.", *([None] * 9))


def build_app():
    with gr.Blocks(title="Transfer Learning Science Lab", delete_cache=(3600, 86400)) as app:
        gr.Markdown("# Transfer Learning Science Lab\nTeach an ImageNet model to recognize bean leaf disease.",
                    elem_id="lab-header")
        gr.Markdown("Choose a few labeled images → reuse frozen visual features → train a new classifier. "
                    "Compare transfer learning with learning from resized pixels on the same images.",
                    elem_classes=["lab-note"])
        state = gr.State(None)
        with gr.Row():
            with gr.Column(elem_id="lab-controls", scale=1):
                gr.Markdown("### 1. Choose your experiment\n**Model:** MobileNetV2, pretrained on ImageNet.\n\n"
                            "**Dataset:** Beans · angular leaf spot, bean rust, healthy.")
                per_class = gr.Slider(5, 100, value=20, step=5, label="Training images per class",
                                     info="Default: only 60 labeled images. Validation always uses 133 different images.")
                epochs = gr.Slider(1, 100, value=30, step=1, label="Training epochs")
                learning_rate = gr.Dropdown([0.0001, 0.001, 0.01, 0.1], value=0.01, label="Learning rate")
                seed = gr.Number(value=42, precision=0, minimum=0, maximum=10000, label="Random seed")
                train_button = gr.Button("Train and compare", variant="primary")
                gr.Markdown("First run downloads about **177 MB** of model weights and data. "
                            "CPU supported; feature extraction can take a few minutes. "
                            "Features are cached for repeated runs.")
                with gr.Accordion("What is being learned?", open=False):
                    gr.Markdown("MobileNetV2 converts each image into 1,280 visual features. Its convolution "
                                "weights and batch normalization statistics stay fixed. We bypass the original "
                                "ImageNet head and train a new three-class linear softmax head. This is "
                                "**feature extraction / linear probing**, a form of transfer learning.\n\n"
                                "The baseline trains the same kind of head on 16 × 16 RGB pixels (768 features). "
                                "Both use the same images, epochs, optimizer and learning rate. Each fits "
                                "normalization on its training features only. This compares representations; "
                                "the baseline is not a deep network trained from scratch.")
            with gr.Column(elem_id="lab-results", scale=2):
                status = gr.Markdown("Train the default experiment, then increase the number of labeled images.")
                with gr.Tabs():
                    with gr.Tab("Compare models"):
                        comparison = gr.Dataframe(label="Training and validation metrics", interactive=False)
                        accuracy_curve = gr.LinePlot(x="Epoch", y="Accuracy", color="Series",
                                                     y_lim=[0, 1], title="Learning curves · accuracy")
                        loss_curve = gr.LinePlot(x="Epoch", y="Loss", color="Series", title="Learning curves · cross-entropy")
                    with gr.Tab("Inspect predictions"):
                        confusion = gr.Dataframe(label="Transfer head confusion matrix · counts", interactive=False)
                        errors = gr.Gallery(label="Validation examples · mistakes first, then correct predictions", columns=3)
                        predictions = gr.Dataframe(label="All validation predictions", interactive=False)
                    with gr.Tab("Training examples"):
                        examples = gr.Gallery(label="Two training images per class", columns=3)
                download = gr.File(label="Download report, learning curves, validation predictions and trained head")
        gr.Markdown("### 2. Try your transferred classifier\nTrain first, then upload a leaf image. "
                    "Scores are model outputs and are not calibrated probabilities of correctness.")
        with gr.Row():
            image = gr.Image(type="pil", sources=["upload"], label="Leaf image")
            scores = gr.Label(label="Transferred leaf classifier", num_top_classes=3)
        predict_button = gr.Button("Classify with the trained head")
        gr.Markdown("### Try it\n1. Compare 5, 20 and 100 training images per class with the same seed. "
                    "Does transfer help when labels are scarce?\n"
                    "2. Increase epochs to 100. Does training accuracy improve more than validation accuracy?\n"
                    "3. Compare learning rates 0.001 and 0.1. Inspect the loss curves.\n"
                    "4. Change the seed. How stable is your conclusion?\n\n"
                    "Validation is used for exploration; its scores are not an independent final test estimate. "
                    "The 128-image official test split is reserved. Performance on this small dataset may not "
                    "generalize to other cameras, farms or leaf conditions.\n\n"
                    f"[Model card](https://huggingface.co/{MODEL_ID}) · "
                    f"[Dataset card (MIT metadata)](https://huggingface.co/datasets/{DATASET_ID}) · "
                    "See the model card for model terms. Downloads occur on first use; training and inference "
                    "run on this server. Uploaded images are used for inference only.")
        settings = [per_class, epochs, learning_rate, seed]
        outputs = [state, status, comparison, accuracy_curve, loss_curve, confusion,
                   predictions, examples, errors, download, scores]
        controls = [*settings, train_button, predict_button, image]
        for control in settings:
            control.input(clear_results, outputs=outputs, queue=False)
        image.input(lambda: None, outputs=scores, queue=False)
        # Lock settings until completion, including errors, to keep results tied to inputs.
        event = train_button.click(
            lambda: [gr.update(interactive=False) for _ in controls] + list(clear_results()),
            outputs=controls + outputs, queue=False,
        )
        event = event.then(run_experiment, settings, outputs,
                           concurrency_id="transfer-model", concurrency_limit=1)
        event.then(lambda: [gr.update(interactive=True) for _ in controls], outputs=controls)
        prediction = predict_button.click(
            lambda: [gr.update(interactive=False) for _ in controls] + [None],
            outputs=controls + [scores], queue=False,
        )
        prediction = prediction.then(predict_image, [image, state], scores,
                                     concurrency_id="transfer-model", concurrency_limit=1)
        prediction.then(lambda: [gr.update(interactive=True) for _ in controls], outputs=controls)
    return app.queue(max_size=20, default_concurrency_limit=1)


if __name__ == "__main__":
    build_app().launch(theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
                       css_paths=[Path(__file__).with_name("styles.css")])
