"""Inference-only Kumo science teaching app: python kumo_app.py."""
from pathlib import Path
from time import perf_counter
from threading import Lock
import tempfile
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report

LOCK = Lock()
MODEL = None
MODEL_SIZE = None
SEPARATORS = {'Comma': ',', 'Semicolon': ';', 'Tab': '\t'}

def read(path, sep):
    if not path:
        raise gr.Error('Upload a CSV first.')
    try:
        frame = pd.read_csv(path, sep=SEPARATORS[sep])
    except Exception as exc:
        raise gr.Error(f'Cannot read CSV: {exc}') from exc
    if frame.empty:
        raise gr.Error('CSV has no rows.')
    return frame

def columns(path, sep):
    df = read(path, sep)
    return gr.Dropdown(choices=list(df), value=df.columns[-1]), gr.Dropdown(choices=list(df), value=[], multiselect=True)

def infer(context, labels, query, size, seed):
    global MODEL, MODEL_SIZE
    try:
        import torch
        import sdm
    except ImportError as exc:
        raise gr.Error('Install requirements-kumo.txt to enable Kumo inference.') from exc
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with LOCK:
        start = perf_counter()
        if MODEL_SIZE != size:
            MODEL = None
            MODEL_SIZE = None
            if device == 'cuda':
                torch.cuda.empty_cache()
            MODEL = sdm.models.KumoTabular(task='classification', size=size, device=device)
            MODEL.eval()
            MODEL_SIZE = size
        load_time = perf_counter() - start
        names = sorted(labels.unique())
        codes = labels.map({v: i for i, v in enumerate(names)})
        features = pd.concat([context, query], ignore_index=True)
        x = sdm.TableTensor.from_pandas(df=features, stypes=sdm.infer_stypes(features), device=device)
        ydf = pd.DataFrame({'target': codes.to_numpy()})
        y = sdm.TableTensor.from_pandas(df=ydf, stypes={'target': 'categorical'}, device=device)
        start = perf_counter()
        torch.manual_seed(int(seed))
        with torch.inference_mode(), torch.amp.autocast(device, dtype=torch.float16, enabled=device == 'cuda'):
            output = MODEL(x_context=x[:len(context)], y_context=y, x_query=x[len(context):], num_estimators=4)
        probabilities = output.to_pandas().reindex(columns=[str(i) for i in range(len(names))]).to_numpy(dtype=float)
        if probabilities.shape != (len(query), len(names)) or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-3):
            raise ValueError('Model returned invalid class probabilities.')
        return np.asarray(names)[probabilities.argmax(1)], probabilities, names, load_time, perf_counter()-start

def clean(df):
    result = df.copy()
    for col in result:
        if pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)
        else:
            result[col] = result[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return result

def export(df):
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as file:
        df.to_csv(file, index=False)
        return file.name

def evaluate(source, path, sep, target, excluded, size, fraction, seed, history):
    if source == 'Wine chemistry':
        data = load_wine(as_frame=True)
        X, labels = data.data, data.target.map(lambda i: f'Cultivar {i+1}')
        dropped = 0
    else:
        df = read(path, sep)
        if target not in df:
            raise gr.Error('Select a target column.')
        dropped = int(df[target].isna().sum())
        df = df.dropna(subset=[target])
        X, labels = df.drop(columns=list(set([target, *(excluded or [])])), errors='ignore'), df[target].astype(str)
    X = clean(X)
    if len(X) > 2500 or not 1 <= X.shape[1] <= 100 or not 2 <= labels.nunique() <= 10:
        raise gr.Error('Use at most 2,500 labeled rows, 1–100 features and 2–10 classes.')
    try:
        dev, test = train_test_split(np.arange(len(X)), test_size=.2, stratify=labels, random_state=int(seed))
        pool, val = train_test_split(dev, test_size=.25, stratify=labels.iloc[dev], random_state=int(seed))
        count = min(1000, int(len(pool)*float(fraction)))
        if count < len(pool):
            pool, _ = train_test_split(pool, train_size=count, stratify=labels.iloc[pool], random_state=int(seed))
        if set(labels.iloc[pool]) != set(labels):
            raise ValueError('Context must contain every class.')
    except ValueError as exc:
        raise gr.Error(f'Cannot stratify this dataset/context. Increase context size or rows per class. {exc}') from exc
    try:
        predicted, probs, names, loading, elapsed = infer(X.iloc[pool], labels.iloc[pool], X.iloc[val], size, seed)
    except Exception as exc:
        raise gr.Error(f'Inference failed; check dependencies/download access, or reduce context/model size: {exc}') from exc
    truth = labels.iloc[val]
    baseline = np.repeat(labels.iloc[pool].mode().iloc[0], len(val))
    score = f1_score(truth, predicted, average='macro')
    summary = f'### Validation results\nAccuracy **{accuracy_score(truth,predicted):.3f}** · Balanced accuracy **{balanced_accuracy_score(truth,predicted):.3f}** · Macro F1 **{score:.3f}** · Majority baseline F1 **{f1_score(truth,baseline,average="macro",zero_division=0):.3f}**\n\nContext: {len(pool)} · Validation: {len(val)} · Hidden test: {len(test)} · Dropped missing targets: {dropped}\n\nCheckpoint loading: {loading:.2f}s · Inference: {elapsed:.2f}s'
    fig, ax = plt.subplots(figsize=(5,4))
    matrix = confusion_matrix(truth, predicted, labels=names)
    ax.imshow(matrix, cmap='Blues')
    for (i,j), value in np.ndenumerate(matrix):
        ax.text(j,i,str(value),ha='center',va='center')
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names, xlabel='Predicted', ylabel='True', title='Validation confusion matrix')
    fig.tight_layout()
    plt.close(fig)
    rows = pd.DataFrame({'Row': X.index[val], 'True': truth.to_numpy(), 'Predicted': predicted})
    for i, name in enumerate(names):
        rows[f'P({name})'] = probs[:,i]
    report = pd.DataFrame(classification_report(truth,predicted,labels=names,output_dict=True,zero_division=0)).T.reset_index(names='Class')
    history = [*(history or []), [source,size,len(pool),int(seed),score,elapsed]][-25:]
    return summary, fig, report, rows, export(rows), history, history, (X.iloc[pool],labels.iloc[pool],size,int(seed))

def predict(state, path, sep):
    if state is None:
        raise gr.Error('Run Predict & evaluate first.')
    context, labels, size, seed = state
    frame = read(path, sep)
    if len(frame)>500 or not set(context).issubset(frame):
        raise gr.Error('Supply all context feature columns and at most 500 query rows.')
    try:
        pred, probs, names, _, _ = infer(context,labels,clean(frame[list(context)]),size,seed)
    except Exception as exc:
        raise gr.Error(f'Inference failed: {exc}') from exc
    result = pd.DataFrame({'Row': frame.index,'Predicted':pred})
    for i,name in enumerate(names):
        result[f'P({name})'] = probs[:,i]
    return result, export(result)

with gr.Blocks(title='Kumo Science Lab') as demo:
    gr.Markdown('# Kumo Science Lab\nScientific classification with a pretrained tabular foundation model.',elem_id='lab-header')
    gr.Markdown('**Labeled context → unlabeled queries → class probabilities.** Weights stay fixed: no training or tuning. Validation labels are used only for scoring; 20% of rows remain hidden for final testing.',elem_classes=['lab-note'])
    state, history = gr.State(None), gr.State([])
    with gr.Row():
        with gr.Column(scale=1,elem_id='lab-controls'):
            source = gr.Dropdown(['Wine chemistry','Uploaded CSV'],value='Wine chemistry',label='Scientific dataset')
            gr.Markdown('**Wine chemistry:** 178 samples, 13 chemical measurements, three cultivars. Predict cultivar from laboratory measurements. Classes are cultivar identifiers, not quality ratings.')
            upload = gr.File(file_types=['.csv','.tsv','.txt'],label='Labeled CSV')
            sep = gr.Dropdown(list(SEPARATORS),value='Comma',label='Separator')
            target = gr.Dropdown([],label='Target')
            excluded = gr.Dropdown([],multiselect=True,label='Exclude IDs or leakage columns')
            size = gr.Radio(['small','medium','large'],value='small',label='Checkpoint size')
            fraction = gr.Radio([.1,.25,.5,1.0],value=1.0,label='Fraction of context pool (capped at 1,000 rows)')
            seed = gr.Dropdown([42,7,123],value=42,label='Random seed')
            run = gr.Button('Predict & evaluate',variant='primary')
        with gr.Column(scale=2,elem_id='lab-results'):
            summary = gr.Markdown('Choose settings, then predict.')
            plot = gr.Plot()
            report = gr.Dataframe(label='Per-class metrics and support',interactive=False)
            rows = gr.Dataframe(label='Inspect validation predictions and probabilities',interactive=False)
            download = gr.File(label='Download validation predictions')
    history_view = gr.Dataframe(headers=['Dataset','Size','Context rows','Seed','Macro F1','Inference seconds'],label='My experiments',interactive=False)
    gr.Markdown('### Predict new CSV\nUses your last successful context and settings. Extra columns, including targets, are ignored.')
    query = gr.File(file_types=['.csv','.tsv','.txt'],label='New query CSV')
    query_sep = gr.Dropdown(list(SEPARATORS),value='Comma',label='Query separator')
    predict_button = gr.Button('Predict new CSV')
    query_rows = gr.Dataframe(interactive=False)
    query_download = gr.File(label='Download new predictions')
    upload.upload(columns,[upload,sep],[target,excluded]).then(lambda:'Uploaded CSV',outputs=source)
    sep.change(columns,[upload,sep],[target,excluded])
    for control in [source,upload,sep,target,excluded,size,fraction,seed]:
        control.change(lambda:(None,'Settings changed. Run Predict & evaluate to refresh results.'),outputs=[state,summary],queue=False)
    run.click(evaluate,[source,upload,sep,target,excluded,size,fraction,seed,history],[summary,plot,report,rows,download,history_view,history,state],concurrency_id='gpu',concurrency_limit=1)
    predict_button.click(predict,[state,query,query_sep],[query_rows,query_download],concurrency_id='gpu',concurrency_limit=1)
    gr.Markdown('### Try it\n1. Hold seed and size fixed; vary context size.\n2. Compare small, medium and large on identical rows. Is larger always better?\n3. Change the seed: how stable is the conclusion?\n4. Inspect a confident mistake. Probabilities are not guaranteed to be calibrated.\n\nRandom row splits assume independent samples; repeated specimens, laboratories and time series need tailored splits. Repeated validation comparisons do not provide an unbiased final test score. Remove IDs and target-derived features.\n\nSources: [Wine dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset) · [NVIDIA Kumo weights](https://huggingface.co/nvidia/Kumo-Tabular) (OpenMDW 1.1).')

demo.queue(max_size=30,default_concurrency_limit=1)
if __name__ == '__main__':
    demo.launch(theme=gr.themes.Base(primary_hue='blue',neutral_hue='slate'),css_paths=[Path(__file__).with_name('styles.css')])
