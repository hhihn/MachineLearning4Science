"""Tabular AutoML: run with `python automl_app.py`.

Preprocessing and hyperparameter selection use training folds only. A held-out
test partition evaluates the selected pipeline after model selection is complete.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from time import perf_counter
import warnings
import zipfile

import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes, load_wine
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

BUILTINS = {
    "Wine chemistry (classification)": ("classification", load_wine),
    "Diabetes progression (regression)": ("regression", load_diabetes),
}
CSV = "Uploaded CSV"
SEPARATORS = {"Comma (,)": ",", "Semicolon (;)": ";", "Tab": "\t", "Pipe (|)": "|"}
BLUE = "#2563a6"
STYLESHEET = Path(__file__).with_name("styles.css")
THEME = gr.themes.Base(primary_hue="blue", secondary_hue="blue", neutral_hue="slate",
                       font=["system-ui", "sans-serif"]).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    button_primary_background_fill_dark="*primary_600",
    button_primary_background_fill_hover_dark="*primary_500",
    button_primary_text_color_dark="white",
)
sns.set_theme(context="paper", style="ticks", font="DejaVu Sans", font_scale=1.15,
              palette=[BLUE], rc={
                  "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
                  "pdf.fonttype": 42, "svg.fonttype": "none", "axes.titleweight": "bold",
                  "axes.titlepad": 12, "axes.spines.top": False, "axes.spines.right": False,
                  "grid.color": "#e6ebf1", "axes.labelcolor": "#24364b",
                  "text.color": "#24364b", "axes.edgecolor": "#9aa9b8",
              })


def display_table(frame):
    return frame.style.format({c: "{:.3f}" for c in frame.select_dtypes("floating")}, na_rep="—")


def model_parameters(state, model_name):
    columns = ["Parameter", "Value", "Selection"]
    if not state or model_name not in state.get("hyperparameters", {}):
        return pd.DataFrame(columns=columns), "Run AutoML to inspect each model's selected parameters."
    entry = state["hyperparameters"][model_name]
    rows = []
    for name, value in sorted(entry["parameters"].items()):
        if isinstance(value, float):
            text = f"{value:.3f}"
        else:
            text = json.dumps(value, ensure_ascii=False)
        rows.append([name, text, "Tuned" if name in entry["tuned"] else "Fixed / default"])
    info = (f"**{model_name}** · Best cross-validation {entry['metric']}: {entry['cv_score']:.3f}. "
            f"Searched {entry['configurations']} configuration(s). "
            "Tuned parameters were chosen by cross-validation; fixed/default parameters were held constant.")
    return pd.DataFrame(rows, columns=columns), info


def initialize_parameters(state):
    names = list(state["hyperparameters"]) if state else []
    selected = state["winner"] if state else None
    table, info = model_parameters(state, selected)
    return gr.Dropdown(choices=names, value=selected), table, info


def read_csv(path, separator):
    if not path:
        raise gr.Error("Upload a CSV file first.")
    try:
        frame = pd.read_csv(path, sep=SEPARATORS[separator])
    except (OSError, ValueError, UnicodeError, KeyError) as exc:
        raise gr.Error(f"Cannot read CSV. Check the separator and UTF-8 encoding. {exc}") from exc
    if frame.empty:
        raise gr.Error("The CSV contains no data rows.")
    return frame


def activate_csv():
    return CSV


def csv_columns(path, separator, current_target):
    if not path:
        return gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=[], multiselect=True), "Upload a CSV and choose its target."
    frame = read_csv(path, separator)
    if frame.shape[1] < 2:
        raise gr.Error("At least two columns are needed. Check the separator.")
    columns = frame.columns.tolist()
    target = current_target if current_target in columns else columns[-1]
    return (gr.Dropdown(choices=columns, value=target), gr.Dropdown(choices=columns, value=[], multiselect=True),
            f"Loaded {len(frame):,} rows · {len(columns):,} columns. Uploaded CSV is available for training.")


def clean_features(frame):
    """Normalize types without learning any statistics from the data."""
    result = frame.copy()
    for column in result:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)
        else:
            result[column] = result[column].map(lambda x: str(x) if pd.notna(x) else np.nan)
    return result


def load_dataset(source, path, separator, target, task, excluded):
    if source in BUILTINS:
        task, loader = BUILTINS[source]
        bunch = loader(as_frame=True)
        return task, clean_features(bunch.data), bunch.target, {"source": source, "dropped_targets": 0}
    if source != CSV:
        raise gr.Error("Choose a dataset.")
    frame = read_csv(path, separator)
    if target not in frame:
        raise gr.Error("Choose the target column from the uploaded CSV.")
    if task not in ("classification", "regression"):
        raise gr.Error("Choose classification or regression.")
    if target in (excluded or []):
        raise gr.Error("The target cannot also be an excluded input column.")
    keep = frame[target].notna()
    y = frame.loc[keep, target].copy()
    X = clean_features(frame.loc[keep].drop(columns=[target, *(excluded or [])], errors="ignore"))
    if X.shape[1] == 0:
        raise gr.Error("Keep at least one input column besides the target.")
    if task == "classification":
        y = y.astype(str)
        if y.nunique() < 2:
            raise gr.Error("Classification requires at least two target classes.")
    else:
        try:
            y = pd.to_numeric(y, errors="raise")
        except (ValueError, TypeError) as exc:
            raise gr.Error("Regression requires a numeric target.") from exc
        if not np.isfinite(y).all():
            raise gr.Error("Regression targets must be finite numbers.")
    return task, X, y, {"source": Path(path).name, "target": target,
                        "dropped_targets": int((~keep).sum()), "excluded_columns": excluded or []}


def preprocessing(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
        ]), categorical),
    ])


def candidate_models(task, seed, min_fold_size):
    classification = task == "classification"
    return {
        "Linear model": (
            LogisticRegression(max_iter=3000, random_state=seed) if classification else Ridge(),
            {"model__C": [0.01, 0.1, 1, 10, 100]} if classification else
            {"model__alpha": [0.01, 0.1, 1, 10, 100]},
        ),
        "Decision tree": (
            DecisionTreeClassifier(random_state=seed) if classification else DecisionTreeRegressor(random_state=seed),
            {"model__max_depth": [3, 6, 10, None], "model__min_samples_leaf": [1, 3, 5]},
        ),
        "Random forest": (
            RandomForestClassifier(n_estimators=70, n_jobs=1, random_state=seed) if classification else
            RandomForestRegressor(n_estimators=70, n_jobs=1, random_state=seed),
            {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 3, 5]},
        ),
        "Extra trees": (
            ExtraTreesClassifier(n_estimators=70, n_jobs=1, random_state=seed) if classification else
            ExtraTreesRegressor(n_estimators=70, n_jobs=1, random_state=seed),
            {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 3, 5]},
        ),
        "Gradient boosting": (
            GradientBoostingClassifier(random_state=seed) if classification else GradientBoostingRegressor(random_state=seed),
            {"model__n_estimators": [40, 70, 100], "model__max_depth": [2, 3],
             "model__learning_rate": [0.03, 0.1]},
        ),
        "K-nearest neighbors": (
            KNeighborsClassifier() if classification else KNeighborsRegressor(),
            {"model__n_neighbors": [k for k in [3, 5, 9, 15] if k <= min_fold_size],
             "model__weights": ["uniform", "distance"]},
        ),
        "Support vector machine": (
            SVC() if classification else SVR(),
            {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.1]},
        ),
        "Dummy baseline": (
            DummyClassifier(strategy="most_frequent") if classification else DummyRegressor(), {},
        ),
    }


def evaluation(task, truth, predictions):
    if task == "classification":
        return {"Accuracy": accuracy_score(truth, predictions),
                "Macro F1": f1_score(truth, predictions, average="macro", zero_division=0),
                "Balanced accuracy": balanced_accuracy_score(truth, predictions)}
    return {"RMSE": float(np.sqrt(mean_squared_error(truth, predictions))),
            "MAE": mean_absolute_error(truth, predictions), "R²": r2_score(truth, predictions)}


def allocate_budget(candidates, budget):
    """Distribute candidate configurations fairly without exceeding finite spaces."""
    capacities = {name: len(ParameterGrid(params)) for name, (_, params) in candidates.items()
                  if name != "Dummy baseline"}
    allocations = dict.fromkeys(capacities, 0)
    remaining = min(int(budget), sum(capacities.values()))
    while remaining:
        for name, capacity in capacities.items():
            if allocations[name] < capacity:
                allocations[name] += 1
                remaining -= 1
                if not remaining:
                    break
    return allocations


def budget_description(budget, folds):
    budget, folds = int(budget), int(folds)
    return (
        f"**Budget: {budget} candidate configurations total**, shared across the seven model families. "
        f"Each configuration is trained on {folds} cross-validation folds. "
        f"The dummy baseline adds one configuration. A complete run performs "
        f"**{(budget + 1) * folds} CV fits plus 8 refits** of the best model in each family and the baseline. "
        "A configuration is one model with one hyperparameter combination. "
        "This limits search work, not elapsed time or money; larger datasets and slower models take longer. "
        "Every family receives at least one configuration, with remaining trials distributed evenly "
        "until its search space is exhausted."
    )


def result_figures(task, leaderboard, folds, y_test, predictions, importance):
    metric = "Macro F1" if task == "classification" else "RMSE"
    ranking, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    order = leaderboard["Model"].tolist()
    sns.boxplot(data=folds, x="Score", y="Model", order=order, color="#c1d9ec",
                width=0.55, showfliers=False, ax=ax)
    sns.stripplot(data=folds, x="Score", y="Model", order=order, color=BLUE,
                  jitter=False, size=4, ax=ax)
    ax.set(xlabel=f"CV {metric} ({'higher' if task == 'classification' else 'lower'} is better)",
           ylabel="", title="Cross-validation · best configuration per model")
    sns.despine(ax=ax, left=True)
    diagnostics, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    if task == "classification":
        labels = np.unique(np.concatenate([np.asarray(y_test), np.asarray(predictions)]))
        sns.heatmap(confusion_matrix(y_test, predictions, labels=labels), annot=True, fmt="d",
                    cmap="Blues", square=True, vmin=0, xticklabels=labels, yticklabels=labels,
                    cbar=False, ax=axes[0])
        axes[0].set(xlabel="Predicted class", ylabel="Observed class", title="Held-out test confusion matrix")
        scores = evaluation(task, y_test, predictions)
        sns.barplot(x=list(scores), y=list(scores.values()), color=BLUE, errorbar=None, ax=axes[1])
        axes[1].set(ylim=(0, 1.12), ylabel="Score", title=f"Held-out test metrics · n = {len(y_test)}")
        axes[1].bar_label(axes[1].containers[0], fmt="%.3f", padding=3)
    else:
        sns.scatterplot(x=y_test, y=predictions, color=BLUE, alpha=0.7, s=30, ax=axes[0])
        low = float(min(np.min(y_test), np.min(predictions)))
        high = float(max(np.max(y_test), np.max(predictions)))
        pad = max((high - low) * 0.05, 1)
        limits = (low - pad, high + pad)
        axes[0].plot(limits, limits, "--", color="#68788a", linewidth=1)
        axes[0].set(xlabel="Observed target", ylabel="Predicted target", xlim=limits, ylim=limits,
                    title="Held-out test predictions")
        axes[0].set_aspect("equal", adjustable="box")
        sns.scatterplot(x=predictions, y=np.asarray(y_test) - predictions, color=BLUE,
                        alpha=0.7, s=30, ax=axes[1])
        axes[1].axhline(0, linestyle="--", color="#68788a", linewidth=1)
        axes[1].set(xlabel="Predicted target", ylabel="Residual (observed − predicted)",
                    title=f"Held-out test residuals · n = {len(y_test)}")
    for axis in axes:
        sns.despine(ax=axis)
    features, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    top = importance.head(10)
    sns.barplot(data=top, x="Importance", y="Feature", color=BLUE, errorbar=None, ax=ax)
    ax.axvline(0, color="#68788a", linewidth=0.7)
    ax.set(xlabel=f"Decrease in {metric}" if task == "classification" else "Increase in RMSE",
           ylabel="", title="Test permutation importance · not causal")
    sns.despine(ax=ax, left=True)
    return ranking, diagnostics, features


def run_automl(source, csv_path, separator, target, csv_task, excluded, folds, budget, seed,
               progress=gr.Progress()):
    seed, folds, budget = int(seed), int(folds), int(budget)
    if not 0 <= seed < 2**32 or not 3 <= folds <= 5 or not 7 <= budget <= 56:
        raise gr.Error("Use 3–5 folds, a budget of 7–56 total configurations, and a seed between 0 and 4294967295.")
    task, X, y, metadata = load_dataset(source, csv_path, separator, target, csv_task, excluded)
    classification = task == "classification"
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y if classification else None)
    except ValueError as exc:
        raise gr.Error(f"Cannot reserve a test split. Add more rows per class or check the task. {exc}") from exc
    if len(X_test) < 2 or len(X_train) < folds * 3:
        raise gr.Error("Too few rows for this fold count. Add data or reduce the number of folds.")
    if classification and y_train.value_counts().min() < folds:
        raise gr.Error(f"Each class needs at least {folds} training rows. Reduce folds or add examples.")
    cv = (StratifiedKFold if classification else KFold)(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(cv.split(X_train, y_train))
    scoring = "f1_macro" if classification else "neg_root_mean_squared_error"
    metric = "Macro F1" if classification else "RMSE"
    candidates = candidate_models(task, seed, min(len(train) for train, _ in splits))
    allocations = allocate_budget(candidates, budget)
    records, fold_records, fitted, failures, notices = [], [], {}, {}, []
    for index, (name, (model, parameters)) in enumerate(candidates.items()):
        progress((index / len(candidates)), desc=f"Searching {name}")
        pipeline = Pipeline([("preprocess", preprocessing(X_train)), ("model", model)])
        search = RandomizedSearchCV(pipeline, parameters, n_iter=allocations.get(name, 1),
                                    scoring=scoring, cv=splits, random_state=seed, n_jobs=1,
                                    refit=True, error_score="raise", return_train_score=True)
        start = perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                search.fit(X_train, y_train)
            if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                notices.append(f"{name}: optimizer did not fully converge; interpret its score cautiously.")
        except (ValueError, TypeError, FloatingPointError) as exc:
            failures[name] = str(exc)
            continue
        best_index = search.best_index_
        raw = search.cv_results_
        direction = 1 if classification else -1
        values = [direction * raw[f"split{i}_test_score"][best_index] for i in range(folds)]
        records.append({"Model": name, f"CV {metric}": float(np.mean(values)),
                        "Fold SD": float(np.std(values)),
                        f"Train {metric}": direction * float(raw["mean_train_score"][best_index]),
                        "Search seconds": perf_counter() - start,
                        "Configurations": len(raw["params"]),
                        "Best parameters": json.dumps(search.best_params_, sort_keys=True)})
        fold_records.extend({"Model": name, "Fold": i + 1, "Score": float(value)} for i, value in enumerate(values))
        fitted[name] = search.best_estimator_
    learned = [row for row in records if row["Model"] != "Dummy baseline"]
    if not learned:
        raise gr.Error(f"No model completed successfully. Details: {failures}")
    leaderboard = pd.DataFrame(records).sort_values(f"CV {metric}", ascending=not classification, kind="stable").reset_index(drop=True)
    leaderboard.insert(0, "Rank", leaderboard[f"CV {metric}"].rank(method="min", ascending=not classification).astype(int))
    winner = next(name for name in leaderboard["Model"] if name != "Dummy baseline")
    pipeline = fitted[winner]  # Already refit on all development rows by RandomizedSearchCV.
    progress(0.9, desc="Evaluating the selected pipeline on the held-out test set")
    predictions = pipeline.predict(X_test)
    test_metrics = evaluation(task, y_test, predictions)
    baseline_metrics = evaluation(task, y_test, fitted["Dummy baseline"].predict(X_test)) if "Dummy baseline" in fitted else {}
    importances = permutation_importance(pipeline, X_test, y_test, scoring=scoring,
                                         n_repeats=5, random_state=seed, n_jobs=1)
    importance = pd.DataFrame({"Feature": X.columns, "Importance": importances.importances_mean,
                               "Shuffle SD": importances.importances_std}).sort_values("Importance", ascending=False)
    fold_frame = pd.DataFrame(fold_records)
    figures = result_figures(task, leaderboard, fold_frame, y_test, predictions, importance)
    winner_row = leaderboard.loc[leaderboard["Model"] == winner].iloc[0]
    summary = (
        f"### Selected pipeline: {winner}\n"
        f"**Cross-validation {metric}:** {winner_row[f'CV {metric}']:.3f} "
        f"(fold SD {winner_row['Fold SD']:.3f}). "
        f"**Held-out test:** " + " · ".join(f"{key}: {value:.3f}" for key, value in test_metrics.items()) +
        f"\n\nSource: {metadata['source']} · Task: {task} · Inputs: {X.shape[1]} · "
        f"Development rows: {len(X_train)} · Test rows: {len(X_test)} · Seed: {seed}. "
        f"Excluded {metadata['dropped_targets']} rows with missing targets.\n\n"
        f"**Search budget:** {budget} candidate configurations across seven model families, "
        "plus one dummy baseline configuration. "
        f"Completed {sum(row['Configurations'] for row in records if row['Model'] != 'Dummy baseline')} "
        "candidate configurations. Budget measures search work rather than elapsed time.\n\n"
        "Models were selected using development cross-validation only; the test set was evaluated afterward. "
        "CV scores reflect hyperparameter selection on these folds and can be optimistic. "
        "Fold SD describes variability across folds, not a confidence interval. "
        "Repeated runs can turn the test set into part of model selection; use fresh external data for a final assessment."
    )
    if baseline_metrics:
        summary += "\n\n**Dummy test baseline:** " + " · ".join(f"{key}: {value:.3f}" for key, value in baseline_metrics.items())
    if failures:
        summary += "\n\n**Skipped models:** " + "; ".join(f"{name}: {reason}" for name, reason in failures.items())
    if notices:
        summary += "\n\n" + " ".join(notices)
    hyperparameters = {}
    for _, row in leaderboard.iterrows():
        name = row["Model"]
        hyperparameters[name] = {
            "parameters": fitted[name].named_steps["model"].get_params(deep=False),
            "tuned": {key.removeprefix("model__"): value for key, value in json.loads(row["Best parameters"]).items()},
            "metric": metric, "cv_score": float(row[f"CV {metric}"]),
            "configurations": int(row["Configurations"]),
        }
    report = {**metadata, "task": task, "seed": seed, "folds": folds,
              "budget_configurations": budget, "budget_unit": "total non-baseline hyperparameter configurations",
              "configurations_per_model": allocations,
              "completed_configurations": {row["Model"]: row["Configurations"] for row in records},
              "development_rows": len(X_train), "test_rows": len(X_test), "features": X.columns.tolist(),
              "winner": winner, "best_parameters": json.loads(winner_row["Best parameters"]),
              "all_model_hyperparameters": hyperparameters,
              "test_metrics": test_metrics, "baseline_test_metrics": baseline_metrics,
              "failed_models": failures, "notices": notices,
              "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    directory = Path(tempfile.mkdtemp(prefix="ml_science_automl_"))
    joblib.dump({"pipeline": pipeline, "report": report}, directory / "pipeline.joblib")
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "analysis.md").write_text(summary, encoding="utf-8")
    leaderboard.to_csv(directory / "leaderboard.csv", index=False)
    fold_frame.to_csv(directory / "cv_folds.csv", index=False)
    importance.to_csv(directory / "feature_importance.csv", index=False)
    pd.DataFrame({"Row index": X_test.index, "Observed": np.asarray(y_test), "Predicted": predictions}).to_csv(directory / "test_predictions.csv", index=False)
    for name, figure in zip(["cross_validation", "test_diagnostics", "feature_importance"], figures):
        figure.savefig(directory / f"{name}.pdf")
        figure.savefig(directory / f"{name}.png", dpi=300)
        plt.close(figure)
    (directory / "README.txt").write_text(
        "Load your own exported artifact with joblib.load('pipeline.joblib').\n"
        "Predict with artifact['pipeline'].predict(dataframe[artifact['report']['features']]).\n"
        "Only load trusted joblib artifacts. Use the package versions recorded in report.json.\n"
        "The pipeline is fitted on development rows; the held-out test rows were not used to fit it.\n",
        encoding="utf-8")
    archive = directory / "automl_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(directory.iterdir()):
            if path != archive:
                bundle.write(path, path.name)
    progress(1, desc="AutoML complete")
    return (summary, display_table(leaderboard.drop(columns="Best parameters")), *figures, str(archive),
            {"pipeline": pipeline, "features": X.columns.tolist(), "task": task,
             "winner": winner, "hyperparameters": hyperparameters})


def predict_csv(state, csv_path, separator):
    if not state:
        raise gr.Error("Run AutoML successfully before predicting on new data.")
    frame = read_csv(csv_path, separator)
    missing = [column for column in state["features"] if column not in frame]
    if missing:
        raise gr.Error(f"Missing required input columns: {', '.join(missing)}")
    X = clean_features(frame[state["features"]])
    result = frame.copy()
    column = "AutoML prediction"
    while column in result:
        column += " (new)"
    result[column] = state["pipeline"].predict(X)
    directory = Path(tempfile.mkdtemp(prefix="ml_science_predictions_"))
    destination = directory / "predictions.csv"
    result.to_csv(destination, index=False)
    return f"Predicted {len(frame):,} rows using the fitted pipeline from the most recent successful run.", str(destination)


with gr.Blocks(title="AutoML Science Lab", delete_cache=(3600, 86400)) as demo:
    gr.Markdown("# AutoML Science Lab\nFrom a scientific dataset to a tuned, evaluated prediction pipeline.", elem_id="lab-header")
    gr.Markdown("Choose data → cross-validate and tune → evaluate on held-out data → download or predict. "
                "Designed for tabular data with independent rows. Remove IDs, future information, and target-derived inputs; "
                "grouped or time-dependent observations need a tailored splitting strategy.", elem_classes=["lab-note"])
    with gr.Row():
        with gr.Column(scale=1, elem_id="lab-controls"):
            source = gr.Dropdown(choices=[*BUILTINS, CSV], value=next(iter(BUILTINS)), label="Dataset")
            with gr.Accordion("Upload your own CSV", open=False):
                upload = gr.File(label="Training dataset", file_types=[".csv", ".tsv", ".txt"], type="filepath")
                separator = gr.Dropdown(choices=list(SEPARATORS), value="Comma (,)", label="Separator")
                target = gr.Dropdown(choices=[], label="Target column")
                task = gr.Radio(choices=["classification", "regression"], value="classification", label="CSV prediction task")
                excluded = gr.Dropdown(choices=[], multiselect=True, label="Exclude input columns (IDs, leakage)")
                status = gr.Markdown("Upload a CSV and choose its target.")
                gr.Markdown("Uploads automatically become active. Built-in datasets determine their own task. "
                            "Missing targets are excluded; input imputation, scaling, and category encoding are fitted within each CV fold. "
                            "Categorical encoding groups infrequent values into at most 30 categories per input.")
                upload.change(activate_csv, outputs=source).then(
                    csv_columns, inputs=[upload, separator, target], outputs=[target, excluded, status])
                separator.change(csv_columns, inputs=[upload, separator, target], outputs=[target, excluded, status])
            folds = gr.Slider(3, 5, step=1, value=3, label="Cross-validation folds")
            budget = gr.Slider(7, 56, step=1, value=21, label="AutoML budget (total candidate configurations)")
            budget_info = gr.Markdown(budget_description(21, 3))
            budget.change(budget_description, inputs=[budget, folds], outputs=budget_info)
            folds.change(budget_description, inputs=[budget, folds], outputs=budget_info)
            seed = gr.Number(value=42, precision=0, label="Random seed")
            gr.Markdown("Seven model families plus a dummy baseline. 80% development / 20% test. "
                        "More configurations and folds increase runtime. Models use one CPU worker.")
            run = gr.Button("Run AutoML", variant="primary")
        with gr.Column(scale=2, elem_id="lab-results"):
            summary = gr.Markdown("Choose a dataset and click **Run AutoML**.")
            leaderboard = gr.Dataframe(label="Cross-validation leaderboard", interactive=False, wrap=True)
            with gr.Tabs():
                with gr.Tab("Model comparison"):
                    comparison = gr.Plot(label="Cross-validation scores")
                with gr.Tab("Test diagnostics"):
                    diagnostics = gr.Plot(label="Held-out test performance")
                with gr.Tab("Feature importance"):
                    importance = gr.Plot(label="Permutation importance")
                    gr.Markdown("Importance measures predictive dependence, not causation. "
                                "Correlated features can share importance; test importance is descriptive and does not drive selection.")
                with gr.Tab("Hyperparameters"):
                    parameter_model = gr.Dropdown(choices=[], label="Model", interactive=True)
                    parameter_info = gr.Markdown("Run AutoML to inspect each model's selected parameters.")
                    parameter_table = gr.Dataframe(headers=["Parameter", "Value", "Selection"],
                                                   datatype=["str", "str", "str"], interactive=False,
                                                   label="Selected model parameters", wrap=True)
                with gr.Tab("Preprocessing"):
                    gr.Markdown(
                        "### How preprocessing is selected\n"
                        "Preprocessing follows **fixed rules based on column type**. AutoML searches "
                        "model hyperparameters; the search budget does not compare preprocessing "
                        "strategies or perform automatic feature selection.\n\n"
                        "| Column type | Steps applied |\n"
                        "| --- | --- |\n"
                        "| Numeric | Replace infinity with missing values → fill missing values with "
                        "the training median → standardize using the training mean and standard deviation. |\n"
                        "| Categorical | Fill missing values with the most frequent training category → "
                        "one-hot encode, grouping infrequent values into at most 30 categories per column. |\n\n"
                        "Column types come from CSV parsing or the built-in dataset. Text columns are "
                        "treated as categorical inputs. These steps apply to every model family, "
                        "including tree models. Completely empty training columns are retained and "
                        "filled with zero. Categories unseen during fitting are ignored by the encoder.\n\n"
                        "### How data leakage is prevented\n"
                        "The 20% test set is reserved before model search. **Imputation values, scaling "
                        "statistics, and category mappings are learned separately inside each CV training "
                        "fold**, then applied to its validation fold. Each model's selected pipeline is "
                        "refitted on the full development partition. The winning pipeline is evaluated "
                        "on the held-out test set afterward; test rows do not fit preprocessing or models.\n\n"
                        "### Preparing your dataset\n"
                        "Rows with missing targets are excluded. All remaining input columns are used "
                        "unless you exclude them in the upload controls. Remove IDs, future information, "
                        "and columns derived from the target. Choose classification for categories or "
                        "regression for numeric outcomes. Random row splits assume independent samples; "
                        "grouped and time-dependent data need a tailored splitting strategy."
                    )
            download = gr.File(label="Download pipeline, report, CSV results and publication figures", interactive=False)
    fitted_state = gr.State(None)
    run.click(run_automl, inputs=[source, upload, separator, target, task, excluded, folds, budget, seed],
              outputs=[summary, leaderboard, comparison, diagnostics, importance, download, fitted_state], concurrency_limit=1).success(
                  initialize_parameters, inputs=fitted_state,
                  outputs=[parameter_model, parameter_table, parameter_info])
    parameter_model.change(model_parameters, inputs=[fitted_state, parameter_model],
                           outputs=[parameter_table, parameter_info])
    gr.Markdown("### Predict on new data\nUses the fitted pipeline from the most recent successful AutoML run. "
                "Upload a CSV with the same input column names; a target column is optional.")
    with gr.Row():
        prediction_upload = gr.File(label="New data CSV", file_types=[".csv", ".tsv", ".txt"], type="filepath")
        prediction_separator = gr.Dropdown(choices=list(SEPARATORS), value="Comma (,)", label="New data separator")
    predict = gr.Button("Generate predictions")
    prediction_status = gr.Markdown()
    prediction_download = gr.File(label="Download predictions", interactive=False)
    predict.click(predict_csv, inputs=[fitted_state, prediction_upload, prediction_separator],
                  outputs=[prediction_status, prediction_download], concurrency_limit=1)

demo.queue(max_size=20, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(theme=THEME, css_paths=[STYLESHEET])
