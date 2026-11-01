"""Shared, low-compute tabular ML explorer for a science workshop.

The web app reports VALIDATION performance only. The reserved test split is kept
out of exploratory model selection; notebooks show final test evaluation.
"""
from __future__ import annotations

from time import perf_counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine, load_diabetes
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

DATASETS = {
    "Wine chemistry (classification)": ("classification", load_wine),
    "Diabetes progression (regression)": ("regression", load_diabetes),
}
ENSEMBLE_NAMES = ["Random forest", "Extra trees", "Gradient boosting"]
TREE_NAMES = ["Decision tree", *ENSEMBLE_NAMES]
MODEL_NAMES = [*TREE_NAMES, "Linear model", "K-nearest neighbors", "Support vector machine"]
CSV_DATASET = "Uploaded CSV"
SEPARATORS = {"Comma (,)": ",", "Semicolon (;)": ";", "Tab": "\t", "Pipe (|)": "|"}
STYLESHEET = Path(__file__).with_name("styles.css")
PLOT_BLUE = "#2563a6"
sns.set_theme(
    context="paper", style="ticks", palette=[PLOT_BLUE], font="DejaVu Sans",
    font_scale=1.15,
    rc={
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlepad": 12,
        "axes.labelcolor": "#24364b", "text.color": "#24364b",
        "axes.edgecolor": "#9aa9b8", "axes.linewidth": 0.7,
        "grid.color": "#e6ebf1", "grid.linewidth": 0.6,
    },
)
APP_THEME = gr.themes.Base(
    primary_hue="blue", secondary_hue="blue", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    button_primary_background_fill_dark="*primary_600",
    button_primary_background_fill_hover_dark="*primary_500",
    button_primary_text_color_dark="white",
)


def read_csv_file(csv_file, separator):
    if not csv_file:
        raise gr.Error("Upload a CSV file first.")
    try:
        frame = pd.read_csv(csv_file, sep=SEPARATORS[separator])
    except (ValueError, OSError, UnicodeError, KeyError) as exc:
        raise gr.Error(f"Could not read CSV. Check the separator and UTF-8 encoding: {exc}") from exc
    if frame.empty or len(frame.columns) < 2:
        raise gr.Error("CSV needs at least two columns and some data. Check the separator.")
    return frame


def format_table(frame):
    """Format floating-point cells for display without rounding source data."""
    return frame.style.format(
        {column: "{:.3f}" for column in frame.select_dtypes(include="floating").columns},
        na_rep="—",
    )


def update_csv_columns(csv_file, separator, current_target=None):
    if not csv_file:
        return gr.Dropdown(choices=[], value=None), "Upload a CSV to select its target column."
    frame = read_csv_file(csv_file, separator)
    columns = frame.columns.tolist()
    target = current_target if current_target in columns else columns[-1]
    return (gr.Dropdown(choices=columns, value=target),
            f"Loaded {len(frame):,} rows and {len(columns)} columns.")


def select_uploaded_dataset():
    # Keep CSV selected on removal or parsing errors so training cannot silently
    # fall back to a built-in dataset.
    return CSV_DATASET


def dataset_frame(dataset_name, csv_file=None, separator="Comma (,)", target_column=None, csv_task="classification"):
    if dataset_name == CSV_DATASET:
        frame = read_csv_file(csv_file, separator)
        if target_column not in frame.columns:
            raise gr.Error("Select a target column from the uploaded CSV.")
        if csv_task not in {"classification", "regression"}:
            raise gr.Error("Select classification or regression.")
        frame = frame.dropna(subset=[target_column])
        X = frame.drop(columns=[target_column]).copy()
        y = frame[target_column].copy()
        if csv_task == "regression":
            try:
                y = pd.to_numeric(y, errors="raise")
            except (ValueError, TypeError) as exc:
                raise gr.Error("Regression requires a numeric target column.") from exc
            if not np.isfinite(y).all():
                raise gr.Error("The regression target must contain finite numbers.")
        else:
            y = y.astype(str)
            if y.nunique() < 2:
                raise gr.Error("Classification requires at least two target classes.")
        for column in X:
            if pd.api.types.is_numeric_dtype(X[column]):
                X[column] = X[column].replace([np.inf, -np.inf], np.nan)
            else:
                X[column] = X[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
        X = X.dropna(axis=1, how="all")
        if X.empty:
            raise gr.Error("CSV needs at least one nonempty input feature besides the target.")
        return csv_task, X, y
    task, loader = DATASETS[dataset_name]
    dataset = loader(as_frame=True)
    return task, dataset.data.drop(columns=[dataset.target.name], errors="ignore"), dataset.target


def make_model(task, name, depth, trees, seed, feature_frame=None):
    if feature_frame is not None:
        numeric = feature_frame.select_dtypes(include="number").columns.tolist()
        categorical = [column for column in feature_frame if column not in numeric]
        preprocessing = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30),
            ), categorical),
        ])
        return make_pipeline(preprocessing, make_model(task, name, depth, trees, seed))
    if task not in {"classification", "regression"}:
        raise ValueError(f"Unknown task: {task}")
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    classification = task == "classification"
    if name == "Linear model":
        estimator = LogisticRegression(max_iter=2000, random_state=int(seed)) if classification else Ridge()
        return make_pipeline(StandardScaler(), estimator)
    if name == "K-nearest neighbors":
        cls = KNeighborsClassifier if classification else KNeighborsRegressor
        return make_pipeline(StandardScaler(), cls(n_neighbors=5))
    if name == "Support vector machine":
        estimator = SVC() if classification else SVR(C=100.0)
        return make_pipeline(StandardScaler(), estimator)
    kwargs = dict(max_depth=int(depth), random_state=int(seed))
    if name == "Decision tree":
        cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
        return cls(**kwargs)
    if name == "Gradient boosting":
        cls = GradientBoostingClassifier if classification else GradientBoostingRegressor
        return cls(n_estimators=int(trees), **kwargs)
    cls = (
        RandomForestClassifier if task == "classification" else RandomForestRegressor
    ) if name == "Random forest" else (
        ExtraTreesClassifier if task == "classification" else ExtraTreesRegressor
    )
    return cls(n_estimators=int(trees), n_jobs=1, **kwargs)


def model_controls(model_name):
    return (
        gr.Slider(interactive=model_name in TREE_NAMES),
        gr.Slider(interactive=model_name in ENSEMBLE_NAMES),
    )


def experiment_split(dataset_name, train_fraction, seed, csv_file=None, separator="Comma (,)", target_column=None, csv_task="classification"):
    """Keep single-model and batch comparisons on exactly the same samples."""
    task, X, y = dataset_frame(dataset_name, csv_file, separator, target_column, csv_task)
    seed = int(seed)
    train_fraction = float(train_fraction)
    strat = y if task == "classification" else None
    try:
        X_dev, _X_test, y_dev, _y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=strat
        )
        # Use a fixed 20% test set, unseen here, and split the remaining 80%.
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, train_size=train_fraction / 0.8,
            random_state=seed, stratify=y_dev if task == "classification" else None,
        )
    except ValueError as exc:
        raise gr.Error(f"Cannot split this dataset. Add more rows per class or select the correct task. {exc}") from exc
    if len(X_train) < 5 or len(X_val) < 2:
        raise gr.Error("Dataset is too small: need at least 5 training and 2 validation rows.")
    return task, X_train, X_val, y_train, y_val, len(_X_test)


def prediction_diagnostics(task, y_true, predicted, labels=None):
    """Diagnostics use validation predictions only; never the reserved test set."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    if task == "classification":
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, predicted, labels=labels, zero_division=0
        )
        data = pd.DataFrame({"Class": [str(label) for label in labels],
                             "Precision": precision, "Recall": recall, "F1": f1})
        sns.barplot(data=data.melt(id_vars="Class", var_name="Metric", value_name="Score"),
                    x="Class", y="Score", hue="Metric",
                    palette=["#173f6b", "#387fba", "#9dc9e5"], errorbar=None, ax=axes[0])
        axes[0].set(ylim=(0, 1.05), title="Validation performance by class")
        axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                       ncol=3, fontsize=8)
        sns.barplot(x=data["Class"], y=support, color=PLOT_BLUE, errorbar=None, ax=axes[1])
        axes[1].bar_label(axes[1].containers[0], fmt="%d", padding=3)
        axes[1].set(xlabel="Class", ylabel="Validation samples", title="Class support")
        axes[1].margins(y=0.15)
    else:
        residuals = np.asarray(y_true) - np.asarray(predicted)
        sns.scatterplot(x=predicted, y=residuals, color=PLOT_BLUE, alpha=0.7,
                        s=30, edgecolor="white", linewidth=0.4, ax=axes[0])
        axes[0].axhline(0, color="#68788a", linestyle="--", linewidth=1)
        axes[0].set(xlabel="Predicted target", ylabel="Residual (observed − predicted)",
                    title="Residuals versus predictions")
        sns.histplot(x=residuals, bins="auto", color=PLOT_BLUE, edgecolor="white", ax=axes[1])
        axes[1].axvline(0, color="#68788a", linestyle="--", linewidth=1)
        axes[1].set(xlabel="Residual (observed − predicted)", ylabel="Validation samples",
                    title="Residual distribution")
    for ax in axes:
        sns.despine(ax=ax)
    return fig


def comparison_diagnostics(results, primary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    scores = results[["Model", primary, f"Train {primary}"]].rename(
        columns={primary: "Validation", f"Train {primary}": "Training"}
    ).melt(id_vars="Model", var_name="Split", value_name=primary)
    sns.scatterplot(data=scores, x=primary, y="Model", hue="Split", style="Split",
                    hue_order=["Training", "Validation"],
                    palette={"Training": "#9aa9b8", "Validation": PLOT_BLUE},
                    s=65, ax=axes[0])
    axes[0].set(title="Training versus validation", ylabel="",
                xlabel=f"{primary} ({'higher' if primary == 'Macro F1' else 'lower'} is better)")
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    axes[0].xaxis.grid(True)
    sns.barplot(data=results, x="Fit seconds", y="Model", color=PLOT_BLUE,
                errorbar=None, ax=axes[1])
    axes[1].set(title="Training cost", ylabel="", xlabel="Fit time (seconds; system dependent)")
    axes[1].xaxis.grid(True)
    for ax in axes:
        ax.set_axisbelow(True)
        sns.despine(ax=ax, left=True)
    return fig


def compare_models(dataset_name, train_fraction, max_depth, n_trees, seed, csv_file=None, separator="Comma (,)", target_column=None, csv_task="classification"):
    task, X_train, X_val, y_train, y_val, test_count = experiment_split(
        dataset_name, train_fraction, seed, csv_file, separator, target_column, csv_task
    )
    classification = task == "classification"
    primary = "Macro F1" if classification else "RMSE"
    rows = []
    baseline = DummyClassifier(strategy="most_frequent") if classification else DummyRegressor()
    candidates = [(name, make_model(task, name, max_depth, n_trees, seed,
                                   X_train if dataset_name == CSV_DATASET else None)) for name in MODEL_NAMES]
    candidates.append(("Dummy baseline", baseline))
    for name, estimator in candidates:
        start = perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = perf_counter() - start
        predicted = estimator.predict(X_val)
        train_predicted = estimator.predict(X_train)
        row = {"Model": name}
        if classification:
            row.update({
                "Macro F1": f1_score(y_val, predicted, average="macro", zero_division=0),
                "Accuracy": accuracy_score(y_val, predicted),
                "Balanced accuracy": balanced_accuracy_score(y_val, predicted),
                "Train Macro F1": f1_score(y_train, train_predicted, average="macro", zero_division=0),
            })
            row["Generalization gap"] = row["Train Macro F1"] - row["Macro F1"]
        else:
            row.update({
                "RMSE": np.sqrt(mean_squared_error(y_val, predicted)),
                "MAE": mean_absolute_error(y_val, predicted),
                "R²": r2_score(y_val, predicted),
                "Train RMSE": np.sqrt(mean_squared_error(y_train, train_predicted)),
            })
            row["Generalization gap"] = row["RMSE"] - row["Train RMSE"]
        row["Fit seconds"] = fit_seconds
        rows.append(row)
    results = pd.DataFrame(rows).sort_values(
        primary, ascending=not classification, kind="stable"
    ).reset_index(drop=True)
    results.insert(0, "Rank", results[primary].rank(method="min", ascending=not classification).astype(int))
    learned = results[results["Model"] != "Dummy baseline"]
    best = learned.iloc[0]
    tied = learned.loc[np.isclose(learned[primary], best[primary], rtol=0, atol=1e-12), "Model"].tolist()
    baseline_score = results.loc[results["Model"] == "Dummy baseline", primary].iloc[0]
    improvement = best[primary] - baseline_score if classification else baseline_score - best[primary]
    summary = (
        f"### Model comparison — {dataset_name}\n"
        f"**Best validation {primary}:** {best[primary]:.3f} ({', '.join(tied)}). "
        f"{'Higher' if classification else 'Lower'} is better. "
        f"The dummy baseline scores {baseline_score:.3f}; improvement over baseline: {improvement:.3f}.\n\n"
        f"Every model uses the same {len(X_train)} training and {len(X_val)} validation samples "
        f"(seed {int(seed)}). Reserved test: {test_count} samples (not used). "
        f"Tree depth: {int(max_depth)}; ensemble trees: {int(n_trees)}.\n\n"
        "A larger positive generalization gap means training performance exceeds validation performance, "
        "which can indicate overfitting. Fit times are approximate and depend on system load. "
        "This is one split: small score differences do not establish a reliable winner. "
        "Repeat with another seed to explore stability."
    )
    if dataset_name == CSV_DATASET:
        summary += f"\n\nCSV: {Path(csv_file).name} · Target: {target_column} · Task: {task}. Rows with missing targets excluded."
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    sns.barplot(
        data=results, y="Model", x=primary, hue="Model", order=results["Model"].tolist(),
        palette={name: "#9aa9b8" if name == "Dummy baseline" else PLOT_BLUE for name in results["Model"]},
        legend=False, errorbar=None, saturation=1, width=0.65, ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xlim(0, max(float(results[primary].max()) * 1.16, 0.01))
    ax.set_ylabel("")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=True)
    ax.set(xlabel=f"Validation {primary} ({'higher' if classification else 'lower'} is better)",
           title=f"Model comparison · validation n = {len(X_val)}")
    return summary, format_table(results), fig, comparison_diagnostics(results, primary)


def run_experiment(dataset_name, model_name, train_fraction, max_depth, n_trees, seed, history, csv_file=None, separator="Comma (,)", target_column=None, csv_task="classification"):
    task, X_train, X_val, y_train, y_val, test_count = experiment_split(
        dataset_name, train_fraction, seed, csv_file, separator, target_column, csv_task)
    seed = int(seed)
    train_fraction = float(train_fraction)
    model = make_model(task, model_name, max_depth, n_trees, seed,
                       X_train if dataset_name == CSV_DATASET else None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    fig, ax = plt.subplots(figsize=(5.6, 4.0), layout="constrained")
    if task == "classification":
        accuracy = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        balanced = balanced_accuracy_score(y_val, y_pred)
        labels = np.unique(pd.concat([y_train, y_val]))
        sns.heatmap(
            confusion_matrix(y_val, y_pred, labels=labels), annot=True, fmt="d",
            cmap="Blues", vmin=0, square=True, linewidths=1, linecolor="white",
            xticklabels=labels, yticklabels=labels, ax=ax,
            cbar_kws={"label": "Samples", "shrink": 0.8},
        )
        ax.set(xlabel="Predicted class", ylabel="Observed class",
               title=f"Validation confusion matrix · n = {len(X_val)}")
        ax.tick_params(axis="both", rotation=0)
        summary = (
            f"### Validation results\n**Accuracy:** {accuracy:.3f}  ·  "
            f"**Macro F1:** {macro_f1:.3f}  ·  **Balanced accuracy:** {balanced:.3f}\n\n"
            f"Train: {len(X_train)} · Validation: {len(X_val)} · "
            f"Reserved test: {test_count} samples (not used)"
        )
        score_text = f"F1={macro_f1:.3f}"
    else:
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        sns.scatterplot(x=y_val, y=y_pred, alpha=0.75, s=32, color=PLOT_BLUE,
                        edgecolor="white", linewidth=0.4, ax=ax)
        low, high = float(min(y_val.min(), y_pred.min())), float(max(y_val.max(), y_pred.max()))
        padding = max((high - low) * 0.05, 1)
        limits = (low - padding, high + padding)
        ax.plot(limits, limits, "--", color="#68788a", linewidth=1, label="Perfect prediction", zorder=0)
        ax.set(xlabel="Observed target", ylabel="Predicted target",
               title=f"Validation predictions · n = {len(X_val)}", xlim=limits, ylim=limits)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
        sns.despine(ax=ax)
        summary = (
            f"### Validation results\n**RMSE:** {rmse:.2f}  ·  "
            f"**MAE:** {mae:.2f}  ·  **R²:** {r2:.3f}\n\n"
            f"Train: {len(X_train)} · Validation: {len(X_val)} · "
            f"Reserved test: {test_count} samples (not used)"
        )
        score_text = f"RMSE={rmse:.3f}"

    if dataset_name == CSV_DATASET:
        summary += f"\n\nCSV: {Path(csv_file).name} · Target: {target_column} · Task: {task}. Rows with missing targets excluded."
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_label = "Impurity-based importance (not causal)"
    else:
        scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"
        importances = permutation_importance(
            model, X_val, y_val, scoring=scoring, n_repeats=5,
            random_state=seed, n_jobs=1,
        ).importances_mean
        importance_label = "Decrease in macro F1" if task == "classification" else "Increase in RMSE"
    features = pd.Series(importances, index=X_train.columns).nlargest(8)
    fig2, ax2 = plt.subplots(figsize=(5.6, 4.0), layout="constrained")
    sns.barplot(x=features.values, y=features.index, color=PLOT_BLUE,
                errorbar=None, saturation=1, width=0.65, ax=ax2)
    ax2.axvline(0, color="#68788a", linewidth=0.7)
    ax2.xaxis.grid(True)
    ax2.set_axisbelow(True)
    ax2.set(xlabel=importance_label, ylabel="",
            title="Top input features" if hasattr(model, "feature_importances_") else "Validation permutation importance · not causal")
    sns.despine(ax=ax2, left=True)

    history = list(history or [])
    history.append([
        (f"CSV: {Path(csv_file).name} → {target_column} ({task})" if dataset_name == CSV_DATASET
         else dataset_name.split(" (")[0]), model_name,
        f"{train_fraction:.0%}",
        int(max_depth) if model_name in TREE_NAMES else None,
        int(n_trees) if model_name in ENSEMBLE_NAMES else None, seed, score_text,
    ])
    history = history[-25:]
    diagnostics = prediction_diagnostics(
        task, y_val, y_pred, labels=model.classes_ if task == "classification" else None
    )
    return summary, fig, fig2, history, history, diagnostics


with gr.Blocks(title="ML Science Lab") as demo:
    gr.Markdown("# ML Science Lab\nExplore tabular machine learning on real scientific datasets. **No coding required.**", elem_id="lab-header")
    gr.Markdown(
        "Change **one setting at a time** and compare validation metrics. "
        "The untouched 20% test set is deliberately hidden; use the notebooks for final evaluation. "
        "These are educational datasets, not tools for clinical decision-making.",
        elem_classes=["lab-note"],
    )
    with gr.Row():
        with gr.Column(scale=1, elem_id="lab-controls"):
            dataset = gr.Dropdown(choices=[*DATASETS, CSV_DATASET], value=list(DATASETS)[0], label="Scientific dataset")
            with gr.Accordion("Upload your own CSV", open=False):
                csv_file = gr.File(label="CSV file", file_types=[".csv", ".tsv", ".txt"], type="filepath")
                separator = gr.Dropdown(choices=list(SEPARATORS), value="Comma (,)", label="Column separator")
                target_column = gr.Dropdown(choices=[], label="Target column")
                csv_task = gr.Radio(choices=["classification", "regression"], value="classification", label="Prediction task")
                csv_status = gr.Markdown("Upload a CSV to select its target column.")
                gr.Markdown(
                    "Uploading a file automatically selects **Uploaded CSV** for experiments. "
                    "Every other column is used as an input: remove IDs and columns that reveal the target. "
                    "Rows with missing targets are excluded. Numeric and categorical inputs are supported; "
                    "missing inputs are imputed and categories encoded using training data only. "
                    "Classification predicts categories; regression predicts a numeric value."
                )
                csv_file.change(select_uploaded_dataset, outputs=dataset).then(
                    update_csv_columns, inputs=[csv_file, separator, target_column],
                    outputs=[target_column, csv_status],
                )
                separator.change(update_csv_columns, inputs=[csv_file, separator, target_column],
                                 outputs=[target_column, csv_status])
            model = gr.Dropdown(choices=MODEL_NAMES, value="Random forest", label="Tabular ML model")
            gr.Markdown(
                "Linear model uses logistic regression for Wine and ridge regression for Diabetes. "
                "Linear, nearest-neighbor, and support vector models standardize inputs using training data. "
                "Tree settings apply only to tree models; other models use fixed defaults."
            )
            train_fraction = gr.Slider(0.4, 0.6, step=0.1, value=0.6, label="Training fraction of full dataset")
            depth = gr.Slider(2, 14, step=1, value=6, label="Maximum tree depth")
            trees = gr.Slider(10, 150, step=10, value=70, label="Number of trees (ensembles)")
            seed = gr.Dropdown(choices=[42, 7, 123], value=42, label="Random seed")
            button = gr.Button("Train & evaluate", variant="primary")
            model.change(model_controls, inputs=model, outputs=[depth, trees])
        with gr.Column(scale=2, elem_id="lab-results"):
            metrics = gr.Markdown("Choose settings and click **Train & evaluate**.")
            with gr.Row():
                plot = gr.Plot(label="Validation predictions")
                importance = gr.Plot(label="Feature importance")
            diagnostics_plot = gr.Plot(label="Validation diagnostics")
            gr.Markdown(
                "**Diagnostics:** For classification, compare precision, recall, and F1 alongside class sample counts. "
                "For regression, residuals should scatter around zero; patterns or a changing spread can reveal model weaknesses."
            )
    gr.Markdown("### My experiments (current browser session)")
    history_view = gr.Dataframe(
        headers=["Dataset", "Model", "Train fraction", "Max depth", "Trees", "Seed", "Validation score"],
        datatype=["str", "str", "str", "number", "number", "number", "str"],
        interactive=False, label="Recorded experiments", wrap=True,
    )
    history_state = gr.State([])
    button.click(
        run_experiment,
        inputs=[dataset, model, train_fraction, depth, trees, seed, history_state,
                csv_file, separator, target_column, csv_task],
        outputs=[metrics, plot, importance, history_view, history_state, diagnostics_plot],
        concurrency_limit=4,
    )
    gr.Markdown(
        "### Compare all models\nRun every model plus a dummy baseline on the selected dataset. "
        "Uses the training fraction, depth, tree count, and seed above, including retained tree settings "
        "when those controls are disabled. Results are ranked by validation Macro F1 or RMSE."
    )
    compare_button = gr.Button("Compare all models", variant="primary")
    comparison_summary = gr.Markdown()
    comparison_table = gr.Dataframe(interactive=False, label="Model comparison", wrap=True)
    comparison_plot = gr.Plot(label="Validation score comparison")
    comparison_diagnostics_plot = gr.Plot(label="Generalization and training cost")
    gr.Markdown(
        "Compare training and validation scores to spot overfitting, then check training cost. "
        "All values describe a single split; the plots do not show uncertainty across repeated runs."
    )
    compare_button.click(
        compare_models,
        inputs=[dataset, train_fraction, depth, trees, seed, csv_file, separator, target_column, csv_task],
        outputs=[comparison_summary, comparison_table, comparison_plot, comparison_diagnostics_plot],
        concurrency_limit=1,
    )
    gr.Markdown(
        "**Try:** Hold the seed fixed and change maximum depth. Then vary the seed. "
        "Compare validation scores, not just training accuracy.\n\n"
        "Datasets: scikit-learn Wine and Diabetes educational datasets. "
        "Source: https://scikit-learn.org/stable/datasets/toy_dataset.html"
    )

demo.queue(max_size=30, default_concurrency_limit=4)

if __name__ == "__main__":
    demo.launch(theme=APP_THEME, css_paths=[STYLESHEET])
