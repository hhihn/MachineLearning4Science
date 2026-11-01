"""Multilingual zero-shot classification: python zeroshot_app.py."""
from pathlib import Path
from threading import Lock
from string import Formatter
from time import perf_counter
import json
import math
import tempfile
import gradio as gr
import pandas as pd

MODEL_ID = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
LOCK = Lock()
CLASSIFIER = None

def validate(text, labels, template):
    text = (text or '').strip()
    names = [line.strip() for line in (labels or '').splitlines() if line.strip()]
    if not text or len(text) > 10000:
        raise gr.Error('Enter text between 1 and 10,000 characters.')
    if not 2 <= len(names) <= 20 or any(len(n)>100 for n in names):
        raise gr.Error('Enter 2–20 labels, one per line, each at most 100 characters.')
    if len({n.casefold() for n in names}) != len(names):
        raise gr.Error('Labels must be distinct, ignoring case.')
    try:
        fields = [(field,spec,conversion) for _,field,spec,conversion in Formatter().parse(template) if field is not None]
        if fields != [('', '', None)]:
            raise ValueError()
        template.format(names[0])
    except (ValueError, KeyError, IndexError, TypeError):
        raise gr.Error('Template must contain exactly one plain {} placeholder.')
    return text, names

def classify(text, labels, mode, threshold, template):
    global CLASSIFIER
    text, names = validate(text, labels, template)
    start = perf_counter()
    try:
        with LOCK:
            import torch
            if CLASSIFIER is None:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device).eval()
                CLASSIFIER = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer, device=device)
            loading = perf_counter()-start
            for name in names:
                size = len(CLASSIFIER.tokenizer(text, template.format(name), truncation=False)['input_ids'])
                if size > 512:
                    raise gr.Error(f'Text plus hypothesis uses {size} tokens; maximum is 512. Shorten the text, label, or template.')
            start = perf_counter()
            with torch.inference_mode():
                output = CLASSIFIER(text, candidate_labels=names, hypothesis_template=template, multi_label=mode=='Multiple labels')
            elapsed = perf_counter()-start
            device = str(CLASSIFIER.device)
            revision = getattr(CLASSIFIER.model.config, '_commit_hash', None)
        ranked = list(zip(output['labels'],map(float,output['scores'])))
        if len(ranked)!=len(names) or set(output['labels'])!=set(names) or any(not math.isfinite(s) or not 0<=s<=1 for _,s in ranked):
            raise ValueError('Model returned invalid scores.')
        ranked.sort(key=lambda row:row[1],reverse=True)
        if mode == 'Single label' and not math.isclose(sum(s for _,s in ranked),1,abs_tol=1e-4):
            raise ValueError('Single-label scores are not normalized.')
        selected = [ranked[0][0]] if mode=='Single label' else [n for n,s in ranked if s>=threshold]
        frame = pd.DataFrame([{'Label':n,'Score':s,'Selected':n in selected} for n,s in ranked])
        report = dict(text=text, candidate_labels=names, template=template, mode=mode, threshold=threshold, scores=dict(ranked), selected_labels=selected, model=MODEL_ID, revision=revision, device=device, loading_seconds=loading, inference_seconds=elapsed)
        with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False,encoding='utf-8') as file:
            json.dump(report,file,ensure_ascii=False,indent=2)
            destination=file.name
        return ('Selected: '+(', '.join(selected) if selected else 'No labels meet the threshold.')+f'\n\n{device} · Loading {loading:.2f}s · Inference {elapsed:.2f}s',frame,frame,destination)
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f'Classification failed. Install requirements-zeroshot.txt, check download access and available memory: {exc}') from exc

with gr.Blocks(title='Zero-Shot Text Science Lab',delete_cache=(3600,86400)) as demo:
    gr.Markdown('# Zero-Shot Text Science Lab\nMultilingual classification with user-defined labels.',elem_id='lab-header')
    gr.Markdown('No task-specific training examples are needed. Scores are not calibrated measures of correctness. Language performance varies.',elem_classes=['lab-note'])
    with gr.Row():
        with gr.Column(elem_id='lab-controls'):
            text=gr.Textbox(label='Text to classify',lines=8)
            labels=gr.Textbox(label='Candidate labels — one per line',lines=6,value='biology\nchemistry\nphysics')
            mode=gr.Radio(['Single label','Multiple labels'],value='Single label',label='Classification mode')
            threshold=gr.Slider(0,1,value=.5,step=.01,label='Multi-label threshold',interactive=False)
            with gr.Accordion('Hypothesis template',open=False):
                template=gr.Textbox(value='This example is {}.',label='Exactly one {} placeholder; translated templates are supported')
            button=gr.Button('Classify text',variant='primary')
        with gr.Column(elem_id='lab-results'):
            status=gr.Markdown('Enter text and candidate labels.')
            table=gr.Dataframe(label='Ranked label scores',interactive=False)
            chart=gr.BarPlot(x='Score',y='Label',title='Label scores',x_lim=[0,1])
            download=gr.File(label='Download full results as JSON')
    gr.Examples([
        ['We measured how temperature changes the rate of a chemical reaction.','biology\nchemistry\nphysics'],
        ['Wir untersuchen, wie Pflanzen auf Trockenheit reagieren.','Biologie\nChemie\nPhysik'],
    ],inputs=[text,labels])
    gr.Markdown('Single-label scores sum to one across your candidates. Multi-label scores are independent and need not sum to one.\n\n### Try it\nChange label wording, add a competing label, or compare single-label and multi-label modes.\n\n[Model card and language limitations](https://huggingface.co/'+MODEL_ID+') · MIT model license. Weights download on first use; inference runs on this server.')
    inputs=[text,labels,mode,threshold,template]
    outputs=[status,table,chart,download]
    for control in inputs:
        control.change(lambda:('Inputs changed. Classify again.',None,None,None),outputs=outputs)
    mode.change(lambda value:gr.Slider(interactive=value=='Multiple labels'),mode,threshold)
    # Keep controls locked until completion, including failures, so stale runs cannot overwrite edits.
    event=button.click(lambda:[gr.update(interactive=False) for _ in range(6)],outputs=[*inputs,button],queue=False)
    event=event.then(classify,inputs,outputs,concurrency_id='text-model',concurrency_limit=1)
    event.then(lambda value:[gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=value=='Multiple labels'),gr.update(interactive=True),gr.update(interactive=True)],mode,[*inputs,button])
demo.queue(max_size=20,default_concurrency_limit=1)
if __name__=='__main__':
    demo.launch(theme=gr.themes.Base(primary_hue='blue',neutral_hue='slate'),css_paths=[Path(__file__).with_name('styles.css')])
