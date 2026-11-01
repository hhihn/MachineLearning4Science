"""Workshop: Textdaten Schritt für Schritt erkunden. Start: python workshop_app.py"""
import csv
import re
from collections import Counter
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="paper", style="whitegrid", font_scale=1.2, palette="deep")
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

EXAMPLE = Path(__file__).parent / 'daten' / 'student_feedback.csv'
EMBED_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
CLASS_MODEL = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'

def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        sample = f.read(8192); f.seek(0)
        try: dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
        except csv.Error: dialect = csv.excel
        rows = list(csv.reader(f, dialect))
    if not rows or not rows[0]: raise gr.Error('Die Datei ist leer.')
    header = rows[0]
    if len(header) == 1:
        rows = [header] + [[dialect.delimiter.join(row)] for row in rows[1:]]
    elif any(len(row) != len(header) for row in rows[1:]):
        raise gr.Error('Die CSV enthält Zeilen mit abweichender Spaltenzahl. Bitte die Datei korrigieren.')
    return pd.DataFrame(rows[1:], columns=header)

def page(df, n):
    n = max(0, min(int(n), max(0, (len(df)-1)//10)))
    return df.iloc[n*10:n*10+10], n, f'Seite {n+1} von {max(1,(len(df)+9)//10)} · {len(df)} Zeilen'

def load(path):
    if not path: raise gr.Error('Bitte zuerst eine CSV-Datei hochladen.')
    try: df = read_csv(path)
    except gr.Error: raise
    except Exception as e: raise gr.Error(f'Datei konnte nicht gelesen werden: {e}')
    table, n, info = page(df, 0)
    return df, gr.update(choices=list(df.columns), value='text' if 'text' in df.columns else df.columns[0]), table, n, info

def texts(df, col):
    if df is None or not col: raise gr.Error('Bitte zuerst in Schritt 1 Daten und Textspalte auswählen.')
    return df[col].fillna('').astype(str).str.strip()

def words(s): return re.findall(r"\b[^\W\d_]+(?:['’][^\W\d_]+)?\b", s.lower(), re.UNICODE)

def explore_texts(df, col):
    series = texts(df, col)
    valid = series[series.ne('')]
    empty = len(series) - len(valid)
    duplicates = int(valid.duplicated().sum())
    lengths = valid.map(lambda text: len(words(text)))
    if lengths.empty:
        raise gr.Error('Die gewählte Spalte enthält keine gültigen Texte.')
    summary = (f'**{len(series)} Rückmeldungen** · **{empty} leer** ({empty/len(series):.1%}) · '
               f'**{duplicates} Duplikate** ({duplicates/len(series):.1%}) · '
               f'Wörter pro gültigem Text: Median **{lengths.median():.1f}**, '
               f'Minimum **{lengths.min()}**, Maximum **{lengths.max()}**.')
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(lengths, discrete=True, color='#4f70b8', ax=ax)
    ax.set_xlabel('Wörter pro Text'); ax.set_ylabel('Anzahl Texte')
    sns.despine(); fig.tight_layout()
    duplicate_examples = valid[valid.duplicated(keep=False)].drop_duplicates().head(10)
    examples = pd.DataFrame({'Doppelte Texte (Beispiele)': duplicate_examples.tolist()})
    return summary, fig, examples

def count_words(df, col):
    s = texts(df,col)
    counts = Counter(w for t in s for w in words(t) if w not in ENGLISH_STOP_WORDS)
    result = pd.DataFrame(counts.most_common(10), columns=['Wort','Anzahl'])
    fig, ax = plt.subplots(figsize=(7, 4.5)); sns.barplot(data=result, x='Anzahl', y='Wort', color='#4f70b8', ax=ax); ax.set_xlabel('Häufigkeit'); ax.set_ylabel(''); sns.despine(); fig.tight_layout()
    return result, fig

def count_pairs(df, col):
    series = texts(df, col)
    counts = Counter()
    for text in series:
        tokens = words(text)
        counts.update((a, b) for a, b in zip(tokens, tokens[1:])
                      if a not in ENGLISH_STOP_WORDS and b not in ENGLISH_STOP_WORDS)
    if not counts: raise gr.Error('Keine Wortpaare nach Entfernen der Stoppwörter gefunden.')
    result = pd.DataFrame([(' '.join(pair), count) for pair, count in counts.most_common(10)], columns=['Wortpaar', 'Anzahl'])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=result, x='Anzahl', y='Wortpaar', color='#4f70b8', ax=ax)
    ax.set_xlabel('Häufigkeit'); ax.set_ylabel(''); sns.despine(); fig.tight_layout()
    return result, fig

def search_terms(df, col, terms):
    series = texts(df, col)
    names = [term.strip() for term in (terms or '').splitlines() if term.strip()]
    if not names: raise gr.Error('Bitte mindestens einen Suchbegriff pro Zeile eingeben.')
    rows = []
    for term in names:
        pattern = re.compile(r'(?<!\w)' + re.escape(term) + r'(?!\w)', re.IGNORECASE)
        counts = series.map(lambda value: len(pattern.findall(value)))
        rows.append({'Suchbegriff': term, 'Vorkommen': int(counts.sum()), 'Texte mit Treffer': int((counts > 0).sum())})
    return pd.DataFrame(rows)

def inference_device():
    """Prefer CUDA, then Apple Silicon GPU, and fall back to CPU."""
    import torch
    if torch.cuda.is_available(): return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device=inference_device())

@lru_cache(maxsize=1)
def classifier():
    from transformers import pipeline
    return pipeline('zero-shot-classification', model=CLASS_MODEL, device=inference_device())

def cluster(df, col, k):
    s = texts(df,col); s = s[s.ne('')]
    k = int(k)
    if len(s) <= k or k < 2: raise gr.Error('Es werden mehr gültige Texte als Cluster benötigt (mindestens zwei Cluster).')
    try:
        x = embedder().encode(s.tolist(), show_progress_bar=False)
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x)
        if len(set(labels)) < 2 or len(set(labels)) >= len(s): raise gr.Error('Für diese Daten sind keine gültigen Clusterkennzahlen berechenbar.')
        xy = PCA(n_components=2).fit_transform(x)
        fig, ax = plt.subplots(figsize=(7, 5)); plot_data = pd.DataFrame({'PCA 1': xy[:,0], 'PCA 2': xy[:,1], 'Cluster': (labels+1).astype(str)}); sns.scatterplot(data=plot_data, x='PCA 1', y='PCA 2', hue='Cluster', palette='deep', s=40, alpha=.8, ax=ax); sns.despine(); fig.tight_layout()
        summary = f'Silhouette: **{silhouette_score(x,labels):.3f}** (höher ist tendenziell besser) · Davies-Bouldin: **{davies_bouldin_score(x,labels):.3f}** (niedriger ist tendenziell besser)'
        result = pd.DataFrame({'Cluster':labels+1,'Text':s.values})
        sizes = result.groupby('Cluster').size().rename('Anzahl').reset_index()
        examples = result.groupby('Cluster',group_keys=False).head(5)
        return summary, fig, sizes, examples
    except gr.Error: raise
    except Exception as e: raise gr.Error(f'Clustering fehlgeschlagen: {e}')

def classic_cluster(df, col, k):
    series = texts(df, col); series = series[series.ne('')]
    k = int(k)
    if len(series) <= k or k < 2: raise gr.Error('Es werden mehr gültige Texte als Cluster benötigt.')
    try:
        matrix = TfidfVectorizer(stop_words='english').fit_transform(series)
        if matrix.shape[1] < 2: raise gr.Error('Für die Darstellung werden mindestens zwei unterschiedliche Wörter benötigt.')
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(matrix)
        if not 1 < len(set(labels)) < len(series): raise gr.Error('Für diese Daten sind keine gültigen Clusterkennzahlen berechenbar.')
        score = silhouette_score(matrix, labels)
        xy = TruncatedSVD(n_components=2, random_state=42).fit_transform(matrix)
        plot_data = pd.DataFrame({'Komponente 1': xy[:,0], 'Komponente 2': xy[:,1], 'Cluster': (labels+1).astype(str)})
        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(data=plot_data, x='Komponente 1', y='Komponente 2', hue='Cluster', palette='deep', s=40, alpha=.8, ax=ax)
        sns.despine(); fig.tight_layout()
        result = pd.DataFrame({'Cluster':labels+1, 'Text':series.values})
        return f'Silhouette: **{score:.3f}**. Die Bewertung basiert auf den TF-IDF-Vektoren.', fig, result.groupby('Cluster').size().rename('Anzahl').reset_index(), result.groupby('Cluster',group_keys=False).head(5)
    except gr.Error: raise
    except Exception as e: raise gr.Error(f'TF-IDF-Clustering fehlgeschlagen: {e}')

def optimize_clusters(df, col, max_k, progress=gr.Progress()):
    s = texts(df, col); s = s[s.ne('')]
    upper = min(int(max_k), len(s)-1)
    if upper < 2: raise gr.Error('Mindestens drei gültige Texte werden benötigt.')
    try:
        import optuna
        x = embedder().encode(s.tolist(), show_progress_bar=False)
        rows = []
        for k in progress.tqdm(range(2, upper+1), desc='Clusterzahlen vergleichen'):
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x)
            if 1 < len(set(labels)) < len(s):
                rows.append({'Clusterzahl': k, 'Silhouette': float(silhouette_score(x, labels))})
        if not rows: raise gr.Error('Für diese Daten konnte keine gültige Clusterlösung berechnet werden.')
        scores = {row['Clusterzahl']: row['Silhouette'] for row in rows}
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.GridSampler({'k': list(scores)}))
        study.optimize(lambda trial: scores[trial.suggest_categorical('k', list(scores))], n_trials=len(scores))
        best = study.best_params['k']
        return gr.update(value=best), pd.DataFrame(rows), f'Empfehlung: **{best} Cluster** (Silhouette {study.best_value:.3f}). Die Zahl wurde für Schritt 5 übernommen; starten Sie das Clustering dort erneut.'
    except gr.Error: raise
    except Exception as e: raise gr.Error(f'Automatische Suche fehlgeschlagen: {e}')

def extract_topics(df, col, minimum):
    series = texts(df, col); series = series[series.ne('')]
    minimum = int(minimum)
    if len(series) < max(10, minimum + 1):
        raise gr.Error('Bitte mindestens zehn bereinigte Texte und mehr Texte als die minimale Themengröße verwenden.')
    try:
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        model = BERTopic(embedding_model=embedder(),
                         umap_model=UMAP(n_neighbors=min(15, len(series)-1), n_components=5, min_dist=0.0, metric='cosine', random_state=42),
                         hdbscan_model=HDBSCAN(min_cluster_size=minimum, metric='euclidean', prediction_data=True),
                         vectorizer_model=CountVectorizer(stop_words='english', ngram_range=(1, 2)), verbose=False)
        docs = series.tolist()
        topic_ids, _ = model.fit_transform(docs)
        rows = []
        for topic in sorted(set(topic_ids)):
            if topic == -1: continue
            keywords = ', '.join(word for word, _ in model.get_topic(topic)[:10])
            examples = model.get_representative_docs(topic) or []
            rows.append({'Thema': topic, 'Anzahl': topic_ids.count(topic), 'Schlüsselwörter': keywords, 'Beispieltexte': ' | '.join(examples[:3])})
        outliers = topic_ids.count(-1)
        result = pd.DataFrame(rows, columns=['Thema', 'Anzahl', 'Schlüsselwörter', 'Beispieltexte'])
        assignments = pd.DataFrame({'Text': docs, 'Thema': ['Nicht zugeordnet' if t == -1 else str(t) for t in topic_ids]})
        status = f'{len(rows)} Themen gefunden; {outliers} von {len(docs)} Texten wurden keinem Thema zugeordnet.'
        return status, result, assignments
    except Exception as e:
        raise gr.Error(f'Themenextraktion fehlgeschlagen: {e}') from e

@lru_cache(maxsize=1)
def sentiment_model():
    from transformers import pipeline
    return pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', device=inference_device())

def sentiment_analysis(df, col, progress=gr.Progress()):
    series = texts(df, col); series = series[series.ne('')]
    if series.empty: raise gr.Error('Keine gültigen bereinigten Texte vorhanden.')
    try:
        model = sentiment_model()
        labels = {'positive': 'positiv', 'negative': 'negativ', 'neutral': 'neutral'}
        records = []
        for text in progress.tqdm(series.tolist(), desc='Stimmung analysieren'):
            result = model(text, truncation=True)[0]
            records.append({'Text': text, 'Sentiment': labels.get(result['label'].lower(), result['label']), 'Score': f"{result['score']:.3f}"})
        frame = pd.DataFrame(records)
        summary = frame.groupby('Sentiment').size().rename('Anzahl').reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=summary, x='Sentiment', y='Anzahl', color='#4f70b8', ax=ax)
        sns.despine(); fig.tight_layout()
        return frame, summary, fig, f'{len(frame)} Texte analysiert. Prüfen Sie die Zuordnungen anhand der Originaltexte.'
    except Exception as e:
        raise gr.Error(f'Sentimentanalyse fehlgeschlagen: {e}') from e

def classify(df, col, labels, minimum, threshold, progress=gr.Progress()):
    s = texts(df,col); names = [x.strip() for x in (labels or '').splitlines() if x.strip()]
    if len(names)<2 or len(set(x.casefold() for x in names)) != len(names): raise gr.Error('Bitte mindestens zwei unterschiedliche Kategorien eingeben.')
    s = s[s.map(lambda x: len(words(x)) >= int(minimum) and bool(x))]
    if s.empty: raise gr.Error('Kein Text erfüllt die Mindestwortzahl.')
    try:
        model = classifier(); records=[]
        for text in progress.tqdm(s.tolist(), desc='Texte klassifizieren'):
            out = model(text, candidate_labels=names, truncation=True)
            score = float(out['scores'][0])
            label = out['labels'][0]
            category = label if score >= threshold else f'unsicher ({label}: {score:.3f})'
            records.append({'Text':text, 'Kategorie':category, 'Score':f'{score:.3f}'})
        result = pd.DataFrame(records)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv',delete=False) as f: path=f.name
        result.to_csv(path,index=False)
        return result, path, f'{len(result)} Texte klassifiziert. Scores sind keine geprüfte Korrektheit.'
    except Exception as e: raise gr.Error(f'Klassifikation fehlgeschlagen: {e}')

WORKSHOP_CSS = """
.gradio-container {max-width: 1120px !important; width: 100% !important; margin-left: auto !important; margin-right: auto !important; background: #f7f8fa !important; color: #172033; font-family: Inter, ui-sans-serif, system-ui, sans-serif;}
#workshop-header {background: #172b4d; color: #f8fafc; padding: 2.5rem 2.75rem; border-radius: 16px; margin: 1rem 0 2rem;}
#workshop-header h1 {color: white !important; font-size: 2.3rem; font-weight: 700; letter-spacing: -.035em; margin-bottom: .35rem;}
#workshop-header h3, #workshop-header p {color: #e1e9f6 !important;}
.step-card {background: #fff; border: 1px solid #e5eaf1; border-radius: 16px; padding: 1.75rem; margin: 1.25rem 0; box-shadow: 0 3px 16px rgba(30, 45, 70, .045);}
.step-card h2 {color: #172b4d; font-size: 1.4rem; font-weight: 700; letter-spacing: -.025em; margin-bottom: .65rem;}
.step-card p {line-height: 1.65;}
.step-card button {border-radius: 8px; font-weight: 600; width: auto !important; min-width: 0 !important; flex: 0 0 auto !important; align-self: flex-start; padding: .5rem 1rem;}
.step-card input, .step-card textarea {border-radius: 9px;}
.step-card table {font-size: .9rem;}
@media (max-width: 600px) { #workshop-header {padding: 1.5rem;} #workshop-header h1 {font-size: 1.8rem;} .step-card {padding: 1rem;} }
"""
with gr.Blocks(title='KI für die Wissenschaft – Textwerkstatt', delete_cache=(3600,86400)) as demo:
    gr.Markdown('# KI für die Wissenschaft\n### Machine Learning mit Texten\nDaten erkunden · bereinigen · Wörter untersuchen · Texte gruppieren · Themen und Kategorien prüfen\n\nWir arbeiten gemeinsam Schritt für Schritt. Modelle werden beim ersten Aufruf heruntergeladen; die Berechnung kann dauern.', elem_id='workshop-header')
    gr.Markdown(f'**Modellberechnung:** {inference_device().upper()} · TF-IDF, Clustering und Dimensionsreduktion laufen auf der CPU.')
    data = gr.State(None); cleaned = gr.State(None); current_page = gr.State(0)
    with gr.Column(elem_classes=["step-card"]):
        gr.Markdown('## 1 · Daten erkunden\nLaden Sie Feedback hoch und verschaffen Sie sich einen Überblick.\n\n**Technisch:** Das Python-Modul `csv` liest die Datei; `pandas` verwaltet die Daten als Tabelle. `Gradio` zeigt jeweils zehn Zeilen an.')
        with gr.Row():
            upload=gr.File(label='CSV hochladen',file_types=['.csv'],type='filepath'); example=gr.Button('Beispieldaten laden')
        column=gr.Dropdown(label='Textspalte auswählen'); info=gr.Markdown(); table=gr.Dataframe(interactive=False,wrap=True)
        with gr.Row(): prev=gr.Button('← Zurück'); next_=gr.Button('Weiter →')
        gr.Markdown('### Datenqualität und Textlängen\nUntersuchen Sie vor der Bereinigung, wie lang die Rückmeldungen sind und wie viele leere oder wiederholte Texte vorliegen. Exakte Duplikate werden nach Entfernen äußerer Leerzeichen gezählt.')
        explore_button = gr.Button('Übersicht berechnen')
        explore_summary = gr.Markdown()
        explore_plot = gr.Plot(label='Verteilung der Textlängen')
        duplicate_table = gr.Dataframe(label='Beispiele für Duplikate', interactive=False, wrap=True)
        explore_button.click(explore_texts, [data,column], [explore_summary,explore_plot,duplicate_table])
        for control in [data,column]: control.change(lambda: (None,None,None), outputs=[explore_summary,explore_plot,duplicate_table])
    with gr.Column(elem_classes=["step-card"]):
        gr.Markdown('## 2 · Daten bereinigen\nNutzen Sie die Übersicht aus Schritt 1, um die Filter zu wählen. Entfernen Sie leere Texte und wählen Sie, ob doppelte Rückmeldungen entfernt werden. Die bereinigten Daten werden für die folgenden Schritte verwendet.\n\n**Technisch:** `pandas` entfernt fehlende und doppelte Werte; Leerzeichen am Textanfang und -ende werden entfernt.')
        deduplicate = gr.Checkbox(label='Doppelte Texte entfernen', value=True)
        with gr.Row():
            length_unit = gr.Radio(['Wörter', 'Zeichen'], value='Wörter', label='Mindestlänge messen in')
            min_length = gr.Number(value=1, minimum=0, precision=0, label='Mindestanzahl')
        clean_button = gr.Button('Daten bereinigen', variant='primary')
        clean_info = gr.Markdown('Bitte zuerst Daten laden und bereinigen.')
        clean_preview = gr.Dataframe(label='Vorschau der bereinigten Texte', interactive=False, wrap=True)
    with gr.Column(elem_classes=["step-card"]):
        gr.Markdown('## 3 · Wörter zählen\nWelche Wörter kommen in den bereinigten Daten häufig vor? Vergleichen Sie einzelne Wörter mit Wortpaaren und überlegen Sie, welche Informationen dabei verloren gehen. Englische Füllwörter werden entfernt. Die Wortzählung berücksichtigt weder Kontext noch Bedeutung.\n\n**Technisch:** `seaborn` gestaltet die Abbildung. Python `re` zerlegt Texte in Wörter, `collections.Counter` zählt sie und die englische Stoppwortliste stammt aus `scikit-learn`. `matplotlib` erstellt das Diagramm.')
        count=gr.Button('Top 10 berechnen',variant='primary'); word_table=gr.Dataframe(interactive=False); word_plot=gr.Plot()
        gr.Markdown('### Häufige Wortpaare\nBenachbarte Wörter zeigen etwas mehr Kontext. Paare mit englischen Stoppwörtern werden ausgelassen. Technisch zählt Python `collections.Counter` die Paare; `seaborn` zeichnet das Diagramm.')
        pair_button = gr.Button('Top 10 Wortpaare berechnen')
        pair_table = gr.Dataframe(label='Häufigste Wortpaare', interactive=False)
        pair_plot = gr.Plot(label='Wortpaare')
        pair_button.click(count_pairs, [cleaned,column], [pair_table,pair_plot])
        for control in [cleaned,column]: control.change(lambda: (None,None), outputs=[pair_table,pair_plot])
        with gr.Accordion('Optional: Eigene Begriffe suchen', open=False):
            gr.Markdown('Geben Sie einen Begriff oder eine Wortgruppe pro Zeile ein. Gezählt werden vollständige Treffer unabhängig von Groß- und Kleinschreibung sowie die Anzahl der Texte mit mindestens einem Treffer. Technisch sucht Python `re` nach den wörtlichen Eingaben.')
            search_input = gr.Textbox(label='Suchbegriffe – einer pro Zeile', lines=3)
            search_button = gr.Button('Begriffe zählen')
            search_results = gr.Dataframe(label='Suchergebnisse', interactive=False)
        search_button.click(search_terms, [cleaned,column,search_input], search_results)
        for control in [cleaned,column,search_input]: control.change(lambda: None, outputs=search_results)
    with gr.Column(elem_classes=["step-card"]):
        gr.Markdown('## 4 · Klassisch vektorisieren und clustern\nTF-IDF beschreibt Texte anhand ihrer Wörter und gewichtet häufige Wörter geringer. Vergleichen Sie diesen Ansatz später mit Embeddings.\n\n**Technisch:** `scikit-learn` berechnet TF-IDF, K-Means und Silhouette; Truncated SVD reduziert die Vektoren für den Plot. `seaborn` visualisiert die Gruppen.')
        classic_k = gr.Slider(2,20,value=3,step=1,label='Anzahl der Cluster')
        classic_button = gr.Button('TF-IDF-Clustering starten',variant='primary')
        classic_info = gr.Markdown()
        classic_plot = gr.Plot()
        classic_sizes = gr.Dataframe(label='Clustergrößen',interactive=False)
        classic_examples = gr.Dataframe(label='Beispieltexte pro Cluster',interactive=False,wrap=True)
        classic_button.click(classic_cluster,[cleaned,column,classic_k],[classic_info,classic_plot,classic_sizes,classic_examples])
        for control in [cleaned,classic_k]: control.change(lambda:('Einstellungen geändert – bitte erneut berechnen.',None,None,None),outputs=[classic_info,classic_plot,classic_sizes,classic_examples])
    with gr.Column(elem_classes=["step-card"]):
        gr.Markdown('## 5 · Ähnliche Texte gruppieren\nEmbeddings bilden Texte als Vektoren ab. K-Means gruppiert sie; PCA zeigt eine vereinfachte zweidimensionale Ansicht. Vergleichen Sie verschiedene Clusterzahlen.\n\n**Technisch:** `sentence-transformers` erzeugt Embeddings mit `paraphrase-multilingual-MiniLM-L12-v2`. `scikit-learn` berechnet K-Means, PCA und die Kennzahlen; `seaborn` und `matplotlib` zeichnen den Plot. Die Kennzahlen verwenden die ursprünglichen Embeddings.')
        k=gr.Slider(2,20,value=3,step=1,label='Anzahl der Cluster'); run_cluster=gr.Button('Embedding und Clustering starten',variant='primary')
        metrics=gr.Markdown(); cluster_plot=gr.Plot(); sizes=gr.Dataframe(label='Clustergrößen',interactive=False); examples=gr.Dataframe(label='Beispieltexte pro Cluster',interactive=False,wrap=True)
    with gr.Column(elem_classes=["step-card"]):
        gr.Markdown('## 6 · Clusterzahl automatisch auswählen\nNachdem Sie in Schritt 5 eine Clusterzahl ausprobiert haben, können Sie diese Auswahl automatisieren. Suchen Sie automatisch eine geeignete Clusterzahl für die Embeddings aus Schritt 5. Vergleichen Sie mehrere Clusterzahlen anhand des Silhouette-Scores. Ein höherer Wert bedeutet hier besser getrennte Gruppen; er garantiert keine inhaltlich sinnvollen Kategorien.\n\n**Technisch:** `sentence-transformers` erzeugt Embeddings, `scikit-learn` berechnet K-Means und Silhouette-Scores. `Optuna` wählt durch eine vollständige Gittersuche die beste getestete Clusterzahl.')
        max_k = gr.Slider(2, 20, value=10, step=1, label='Größte zu prüfende Clusterzahl')
        auto_button = gr.Button('Clusterzahlen automatisch vergleichen', variant='primary')
        auto_info = gr.Markdown()
        auto_table = gr.Dataframe(label='Vergleich der Clusterzahlen', interactive=False)
    with gr.Column(elem_classes=["step-card"]):
        gr.Markdown('## 7 · Themen entdecken mit BERTopic\nFinden Sie Themen in den bereinigten Rückmeldungen. Lesen Sie Schlüsselwörter und Beispiele gemeinsam und überlegen Sie passende Kategorien für Schritt 9. Nicht zugeordnete Texte werden gesondert ausgewiesen.\n\n**Technisch:** `BERTopic` nutzt die mehrsprachigen Sentence-Embeddings, `UMAP` zur Dimensionsreduktion, `HDBSCAN` zur Gruppierung und c-TF-IDF für charakteristische Wörter und Wortpaare. Englische Stoppwörter werden entfernt. Ergebnisse sind explorative Vorschläge und können je nach Einstellungen variieren.')
        topic_minimum = gr.Slider(2,50,value=10,step=1,label='Minimale Themengröße')
        topic_button = gr.Button('Themen extrahieren',variant='primary')
        topic_status = gr.Markdown()
        topic_table = gr.Dataframe(label='Themen, Schlüsselwörter und Beispiele',interactive=False,wrap=True)
        topic_assignments = gr.Dataframe(label='Thematische Zuordnung pro Text',interactive=False,wrap=True)
        topic_button.click(extract_topics,[cleaned,column,topic_minimum],[topic_status,topic_table,topic_assignments])
        for control in [cleaned,topic_minimum]: control.change(lambda: ('Einstellungen geändert – bitte erneut berechnen.',None,None),outputs=[topic_status,topic_table,topic_assignments])
    with gr.Column(elem_classes=["step-card"]):
        gr.Markdown('## 8 · Stimmung analysieren\nUntersuchen Sie, ob die bereinigten Rückmeldungen eher positiv, neutral oder negativ formuliert sind. Das Modell wurde auf englischen Social-Media-Texten trainiert; bei studentischem Feedback und anderen Sprachen kann es sich irren. Scores sind keine kalibrierten Wahrscheinlichkeiten.\n\n**Technisch:** Die `transformers`-Pipeline nutzt das Modell `cardiffnlp/twitter-roberta-base-sentiment-latest` mit `PyTorch`. `pandas` fasst Ergebnisse zusammen und `seaborn` visualisiert sie.')
        sentiment_button = gr.Button('Sentimentanalyse starten', variant='primary')
        sentiment_status = gr.Markdown()
        sentiment_summary = gr.Dataframe(label='Anzahl pro Stimmung', interactive=False)
        sentiment_plot = gr.Plot(label='Verteilung der Stimmung')
        sentiment_results = gr.Dataframe(label='Ergebnisse pro Text', interactive=False, wrap=True)
        sentiment_button.click(sentiment_analysis, [cleaned,column], [sentiment_results,sentiment_summary,sentiment_plot,sentiment_status])
        cleaned.change(lambda: (None,None,None,'Daten geändert – bitte erneut analysieren.'), outputs=[sentiment_results,sentiment_summary,sentiment_plot,sentiment_status])
    with gr.Column(elem_classes=["step-card"]):
        gr.Markdown('## 9 · Eigene Kategorien anwenden\nNutzen Sie die in Schritt 7 gefundenen Themen, um eigene Kategorien zu formulieren. Vergleichen Sie anschließend die Klassifikation mit den Beispielen und mit der Stimmung aus Schritt 8. Zero-Shot ordnet Texte ohne aufgabenspezifische Trainingsbeispiele zu. Prüfen Sie die Ergebnisse kritisch.\n\n**Technisch:** Die `transformers`-Pipeline für Zero-Shot-Klassifikation nutzt das Modell `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` mit `PyTorch`. `pandas` erstellt die Ergebnistabelle und den CSV-Export.')
        categories=gr.Textbox(label='Kategorien – eine pro Zeile',lines=4,value='teaching\nfacilities\nexaminations'); minimum=gr.Slider(1,100,value=5,step=1,label='Mindestanzahl Wörter pro Text')
        confidence=gr.Slider(0,1,value=0.5,step=0.01,label='Mindestscore für eine Zuordnung',info='Unterhalb dieses Werts: unsicher mit bestpassendem Label und Score. Der Score ist keine kalibrierte Wahrscheinlichkeit.')
        run_class=gr.Button('Texte klassifizieren',variant='primary'); status=gr.Markdown(); results=gr.Dataframe(interactive=False,wrap=True); download=gr.File(label='Ergebnisse als CSV')
    load_outputs=[data,column,table,current_page,info]
    upload.change(load,upload,load_outputs); example.click(lambda:load(str(EXAMPLE)),outputs=load_outputs)
    def move(df,n,delta):
        if df is None: raise gr.Error('Bitte zuerst Daten laden.')
        return page(df,n+delta)
    prev.click(lambda d,n:move(d,n,-1),[data,current_page],[table,current_page,info]); next_.click(lambda d,n:move(d,n,1),[data,current_page],[table,current_page,info])
    def clean_data(df, col, remove_duplicates, unit, minimum):
        series = texts(df, col)
        valid = series.ne('')
        result = df.loc[valid].copy()
        result[col] = series[valid]
        empty = int((~valid).sum())
        minimum = int(minimum or 0)
        if minimum < 0: raise gr.Error('Die Mindestanzahl darf nicht negativ sein.')
        lengths = result[col].map(lambda text: len(words(text)) if unit == 'Wörter' else len(text))
        too_short = int((lengths < minimum).sum())
        result = result.loc[lengths >= minimum].copy()
        before = len(result)
        if remove_duplicates: result = result.drop_duplicates(subset=[col])
        if result.empty: raise gr.Error('Nach der Bereinigung sind keine Texte übrig.')
        return result, f'{len(result)} Texte verbleiben · {empty} leere Texte, {too_short} zu kurze Texte und {before-len(result)} Duplikate entfernt.', result.head(10)
    clean_button.click(clean_data, [data,column,deduplicate,length_unit,min_length], [cleaned,clean_info,clean_preview])
    auto_button.click(optimize_clusters, [cleaned,column,max_k], [k,auto_table,auto_info])
    cleaned.change(lambda:(None,'Bereinigte Daten geändert – bitte erneut suchen.'), outputs=[auto_table,auto_info])
    max_k.change(lambda:(None,'Suchbereich geändert – bitte erneut suchen.'), outputs=[auto_table,auto_info])
    count.click(count_words,[cleaned,column],[word_table,word_plot]); run_cluster.click(cluster,[cleaned,column,k],[metrics,cluster_plot,sizes,examples]); run_class.click(classify,[cleaned,column,categories,minimum,confidence],[results,download,status])
    def clear(): return None,None,'Einstellungen geändert – bitte erneut berechnen.',None,None,None,None,None,'Einstellungen geändert – bitte erneut klassifizieren.'
    for control in [data,column]:
        control.change(clear,outputs=[word_table,word_plot,metrics,cluster_plot,sizes,examples,results,download,status])
        control.change(lambda:(None,'Daten geändert – bitte erneut bereinigen.',None),outputs=[cleaned,clean_info,clean_preview])
    for control in [deduplicate, length_unit, min_length]:
        control.change(lambda:(None,'Einstellung geändert – bitte erneut bereinigen.',None),outputs=[cleaned,clean_info,clean_preview])
    cleaned.change(lambda:(None,None), outputs=[word_table,word_plot])
    cleaned.change(lambda:('Bereinigte Daten geändert – bitte erneut berechnen.',None,None,None,None,None,'Bereinigte Daten geändert – bitte erneut klassifizieren.'),outputs=[metrics,cluster_plot,sizes,examples,results,download,status])
    k.change(lambda:('Clusterzahl geändert – bitte erneut berechnen.',None,None,None),outputs=[metrics,cluster_plot,sizes,examples])
    for control in [categories,minimum,confidence]: control.change(lambda:(None,None,'Einstellungen geändert – bitte erneut klassifizieren.'),outputs=[results,download,status])
demo.queue(default_concurrency_limit=1)
if __name__ == '__main__': demo.launch(theme=gr.themes.Base(primary_hue='blue',neutral_hue='slate'),css=WORKSHOP_CSS)
