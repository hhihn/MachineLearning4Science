"""Tabellarische Klassifikation im Workshop: python classification_workshop_app.py"""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

URL='https://huggingface.co/datasets/jason1966/abhinand05_magic-gamma-telescope-dataset/resolve/main/telescope_data.csv'
sns.set_theme(style='whitegrid',context='paper',font_scale=1.15)
CSS='''.gradio-container {max-width:1120px!important;margin:auto!important;background:#f7f8fa!important} #hero {background:#172b4d;color:white;padding:2.5rem;border-radius:16px} #hero h1,#hero p {color:white!important} .step {background:white;border:1px solid #e5eaf1;border-radius:16px;padding:1.75rem;margin:1.25rem 0;box-shadow:0 3px 16px #1e2d460b} .step button {width:auto!important;min-width:0!important;flex:0 0 auto!important;align-self:flex-start;border-radius:8px}'''

def display_table(frame):
    """Format floating-point cells for display without changing model data or exports."""
    return frame.map(lambda value: f'{value:.2f}' if isinstance(value, (float, np.floating)) and pd.notna(value) else value)

def load_data(path=None):
    try: df=pd.read_csv(path or URL)
    except Exception as e: raise gr.Error(f'Daten konnten nicht geladen werden: {e}')
    if df.empty: raise gr.Error('Die Datei ist leer.')
    cols=list(df.columns); target='class' if 'class' in cols else cols[-1]
    features=[c for c in cols if c!=target and not c.startswith('Unnamed:')]
    return df, gr.update(choices=cols,value=target),gr.update(choices=cols,value=features),display_table(df.head(10)),0,f'{len(df)} Zeilen · Seite 1 von {(len(df)+9)//10}'

def page(df,n,delta):
    if df is None: raise gr.Error('Bitte zuerst Daten laden.')
    n=max(0,min(n+delta,(len(df)-1)//10)); return display_table(df.iloc[n*10:n*10+10]),n,f'{len(df)} Zeilen · Seite {n+1} von {(len(df)+9)//10}'

def inspect(df,target):
    if df is None or target not in df: raise gr.Error('Bitte Daten und Zielspalte auswählen.')
    info=pd.DataFrame({'Spalte':df.columns,'Typ':df.dtypes.astype(str).values,'Fehlend':df.isna().sum().values,'Eindeutig':df.nunique().values})
    counts=df[target].value_counts(dropna=False).rename_axis('Klasse').reset_index(name='Anzahl')
    return display_table(info),display_table(counts),f'{len(df)} Zeilen · {df.duplicated().sum()} exakt doppelte Zeilen'

def scaling_preview(df, target, scaling):
    if df is None or target not in df: return None
    numeric = df.drop(columns=target).select_dtypes(include=np.number)
    if numeric.empty: return pd.DataFrame(columns=['Merkmal','Mittelwert vorher','Std. vorher','Mittelwert nachher','Std. nachher','Minimum nachher','Maximum nachher'])
    scaler = {'Standardisierung (z-Score)': StandardScaler(), 'Normalisierung (Min–Max)': MinMaxScaler(), 'Keine Skalierung': 'passthrough'}[scaling]
    pipeline = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), scaler)
    result = pd.DataFrame(pipeline.fit_transform(numeric), columns=numeric.columns)
    return display_table(pd.DataFrame({'Merkmal': numeric.columns, 'Mittelwert vorher': numeric.mean().values, 'Std. vorher': numeric.std(ddof=0).values, 'Mittelwert nachher': result.mean().values, 'Std. nachher': result.std(ddof=0).values, 'Minimum nachher': result.min().values, 'Maximum nachher': result.max().values}))

def clean(df,target,features,dedup):
    if df is None or target not in df or not features or target in features: raise gr.Error('Bitte Zielspalte und mindestens ein anderes Merkmal auswählen.')
    d=df[[target]+features].dropna(subset=[target]).copy(); before=len(d)
    if dedup: d=d.drop_duplicates()
    if d[target].nunique()!=2: raise gr.Error('Diese Anwendung benötigt genau zwei Zielklassen.')
    numeric = d[features].select_dtypes(include=np.number)
    stats = pd.DataFrame({'Merkmal': numeric.columns, 'Mittelwert': numeric.mean().values, 'Standardabweichung': numeric.std().values})
    return d,f'{len(d)} Zeilen nach Bereinigung; {len(df)-before} ohne Zielwert und {before-len(d)} Duplikate entfernt.',gr.update(choices=features,value=features[0]),gr.update(choices=features,value=features[min(1,len(features)-1)]),display_table(stats)

def explore(df,target,features,x,y):
    if df is None: raise gr.Error('Bitte zuerst Daten laden.')
    if not features or target in features or not x or not y or x not in features or y not in features or target not in df: raise gr.Error('Bitte gültige Merkmale, Zielspalte und Achsen auswählen.')
    df = df[[target] + list(features)]
    summary=df.describe(include='all').transpose().reset_index().rename(columns={'index':'Merkmal'})
    fig,ax=plt.subplots(figsize=(7,4)); sns.scatterplot(data=df,x=x,y=y,hue=target,alpha=.45,s=20,ax=ax); fig.tight_layout()
    numeric=df[list(features)].select_dtypes(include='number'); fig2,ax2=plt.subplots(figsize=(8,6)); sns.heatmap(numeric.corr(),center=0,cmap='vlag',ax=ax2);fig2.tight_layout()
    return display_table(summary),fig,fig2

def split_data(df,target,scaling,test_fraction):
    if df is None: raise gr.Error('Bitte zuerst Daten bereinigen.')
    try:
        X=df.drop(columns=target); y=df[target].astype(str); classes=sorted(y.unique()); positive='g' if 'g' in classes else classes[-1]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=float(test_fraction),random_state=42,stratify=y)
        num=X.select_dtypes(include=np.number).columns.tolist(); cat=[c for c in X.columns if c not in num]
        scaler = {'Standardisierung (z-Score)': StandardScaler(), 'Normalisierung (Min–Max)': MinMaxScaler(), 'Keine Skalierung': 'passthrough'}[scaling]
        pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),scaler),num),('cat',make_pipeline(SimpleImputer(strategy='constant',fill_value='missing'),OneHotEncoder(handle_unknown='ignore')),cat)])
        if num:
            numeric_pipeline = clone(pre.named_transformers_['num']) if hasattr(pre, 'named_transformers_') else clone(pre.transformers[0][1])
            transformed = numeric_pipeline.fit_transform(Xtr[num])
            transformed = pd.DataFrame(transformed, columns=num)
            before = Xtr[num]
            stats = pd.DataFrame({'Merkmal': num, 'Mittelwert vorher': before.mean().values, 'Std. vorher': before.std(ddof=0).values, 'Mittelwert nachher': transformed.mean().values, 'Std. nachher': transformed.std(ddof=0).values, 'Minimum nachher': transformed.min().values, 'Maximum nachher': transformed.max().values})
        else:
            stats = pd.DataFrame(columns=['Merkmal','Mittelwert vorher','Std. vorher','Mittelwert nachher','Std. nachher','Minimum nachher','Maximum nachher'])
        return {'Xtr':Xtr,'Xte':Xte,'ytr':ytr,'yte':yte,'pre':pre,'positive':positive,'models':{}},f'{len(Xtr)} Trainings- und {len(Xte)} Testzeilen · positive Klasse: {positive}',display_table(stats)
    except Exception as e: raise gr.Error(f'Trainingsaufteilung fehlgeschlagen: {e}')

def fit_models(state,kind):
    if state is None: raise gr.Error('Bitte zuerst die Trainingsaufteilung erstellen.')
    from sklearn.base import clone
    state=dict(state); state['models']=dict(state['models'])
    models={'Mehrheitsklasse':DummyClassifier(strategy='most_frequent'),'Logistische Regression':LogisticRegression(max_iter=1000),'Random Forest':RandomForestClassifier(n_estimators=150,min_samples_leaf=2,random_state=42,n_jobs=1)}
    for name in (['Mehrheitsklasse','Logistische Regression'] if kind=='baseline' else ['Random Forest']):
        m=make_pipeline(clone(state['pre']),models[name]);m.fit(state['Xtr'],state['ytr']);state['models'][name]=m
    # Training results are exploratory; final evaluation is explicitly triggered later.
    return state,gr.update(choices=list(state['models']),value=list(state['models'])[-1]),f"Trainiert: {', '.join(state['models'])}. Testauswertung im jeweiligen Modellblock."

def train_baseline_with_evaluation(state):
    if state is None: raise gr.Error('Bitte zuerst die Trainingsaufteilung erstellen.')
    from sklearn.base import clone
    try:
        state, model_choice, message = fit_models(state, 'baseline')
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        rows = []
        for name in ['Mehrheitsklasse', 'Logistische Regression']:
            estimator = clone(state['models'][name])
            f1 = cross_val_score(estimator, state['Xtr'], state['ytr'], cv=cv, scoring='f1_macro', n_jobs=1)
            auc = cross_val_score(estimator, state['Xtr'], state['ytr'], cv=cv, scoring='roc_auc', n_jobs=1)
            rows.append({'Modell': name, 'Makro-F1 (CV)': round(float(f1.mean()), 3), 'ROC-AUC (CV)': round(float(auc.mean()), 3)})
        return state, model_choice, message, display_table(pd.DataFrame(rows))
    except Exception as e:
        raise gr.Error(f'Training oder Kreuzvalidierung fehlgeschlagen: {e}') from e

def tune(state,trials,progress=gr.Progress()):
    if state is None: raise gr.Error('Bitte zuerst die Trainingsaufteilung erstellen.')
    import optuna
    from sklearn.base import clone
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    def objective(t):
        m=make_pipeline(clone(state['pre']),RandomForestClassifier(n_estimators=100,max_depth=t.suggest_int('max_depth',3,16),min_samples_leaf=t.suggest_int('min_samples_leaf',1,10),random_state=42,n_jobs=1))
        return cross_val_score(m,state['Xtr'],state['ytr'],cv=cv,scoring='roc_auc',n_jobs=1).mean()
    try:
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.RandomSampler(seed=42));study.optimize(objective,n_trials=int(trials))
        model=make_pipeline(clone(state['pre']),RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=1,**study.best_params));model.fit(state['Xtr'],state['ytr'])
        state=dict(state);state['models']=dict(state['models']);state['models']['Optimierter Random Forest']=model
        table=pd.DataFrame([{'Versuch':t.number+1,'CV ROC-AUC':round(t.value,3),**t.params} for t in study.trials if t.value is not None])
        return state,display_table(table),gr.update(choices=list(state['models']),value='Optimierter Random Forest')
    except Exception as e: raise gr.Error(f'Suche fehlgeschlagen: {e}')

def automl_search(state, budget, progress=gr.Progress()):
    if state is None: raise gr.Error('Bitte zuerst die Trainingsaufteilung erstellen.')
    if state['ytr'].value_counts().min() < 3: raise gr.Error('Jede Klasse benötigt mindestens drei Trainingsbeispiele.')
    candidates = {
        'Logistische Regression': (LogisticRegression(max_iter=3000, random_state=42), {'model__C':[.01,.1,1,10,100]}),
        'Entscheidungsbaum': (DecisionTreeClassifier(random_state=42), {'model__max_depth':[3,6,10,None], 'model__min_samples_leaf':[1,3,5]}),
        'Random Forest': (RandomForestClassifier(n_estimators=70,random_state=42,n_jobs=1), {'model__max_depth':[4,8,None], 'model__min_samples_leaf':[1,3,5]}),
        'Extra Trees': (ExtraTreesClassifier(n_estimators=70,random_state=42,n_jobs=1), {'model__max_depth':[4,8,None], 'model__min_samples_leaf':[1,3,5]}),
        'Gradient Boosting': (GradientBoostingClassifier(random_state=42), {'model__n_estimators':[40,70,100], 'model__max_depth':[2,3], 'model__learning_rate':[.03,.1]}),
        'Nächste Nachbarn': (KNeighborsClassifier(), {'model__n_neighbors':[3,5,9], 'model__weights':['uniform','distance']}),
        'Support Vector Machine': (SVC(probability=True,random_state=42), {'model__C':[.1,1,10,100], 'model__gamma':['scale',.01,.1]}),
    }
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    records=[]; fitted={}; failures=[]
    for index,(name,(estimator,params)) in enumerate(candidates.items()):
        progress(index/len(candidates),desc=f'Suche: {name}')
        n=int(budget)//len(candidates)+(1 if index<int(budget)%len(candidates) else 0)
        try:
            search=RandomizedSearchCV(Pipeline([('preprocess',clone(state['pre'])),('model',estimator)]),params,n_iter=n,scoring='f1_macro',cv=cv,random_state=42,n_jobs=1,refit=True,error_score='raise',return_train_score=True)
            search.fit(state['Xtr'],state['ytr'])
            records.append({'Modell':name,'CV Makro-F1':float(search.best_score_),'Streuung':float(search.cv_results_['std_test_score'][search.best_index_]),'Konfigurationen':n,'Beste Parameter':str(search.best_params_)})
            fitted[name]=search.best_estimator_
        except Exception as e: failures.append(f'{name}: {e}')
    if not records: raise gr.Error('Kein Modell konnte trainiert werden: '+'; '.join(failures))
    leaderboard=pd.DataFrame(records).sort_values('CV Makro-F1',ascending=False).reset_index(drop=True)
    winner=leaderboard.iloc[0]['Modell']
    updated=dict(state);updated['models']=dict(state['models']);updated['models']['AutoML: bestes Modell']=fitted[winner]
    fig,ax=plt.subplots(figsize=(7,4));sns.barplot(data=leaderboard,x='CV Makro-F1',y='Modell',color='#4f70b8',ax=ax);ax.set_xlim(0,1);fig.tight_layout()
    message=f'Bestes Modell: **{winner}** · Auswahl nach mittlerem Makro-F1 der Trainings-Kreuzvalidierung.'
    if failures: message+=' Nicht erfolgreich: '+'; '.join(failures)
    return updated,display_table(leaderboard),fig,message,gr.update(choices=list(updated['models']),value='AutoML: bestes Modell')

def selected(state,name):
    if not state or name not in state['models']: raise gr.Error('Bitte zuerst ein Modell trainieren und auswählen.')
    return state['models'][name]

def evaluate(state,name,threshold):
    model=selected(state,name);X=state['Xte'];y=state['yte'];positive=state['positive'];negative=next(c for c in model.classes_ if c!=positive)
    prob=model.predict_proba(X)[:,list(model.classes_).index(positive)]; pred=np.where(prob>=threshold,positive,negative);truth=(y==positive).astype(int)
    report=pd.DataFrame(classification_report(y,pred,output_dict=True,zero_division=0)).transpose().round(3).reset_index(names='Klasse')
    fig,axes=plt.subplots(1,3,figsize=(15,4));sns.heatmap(confusion_matrix(y,pred,labels=[negative,positive]),annot=True,fmt='d',xticklabels=[negative,positive],yticklabels=[negative,positive],ax=axes[0],cmap='Blues');axes[0].set(xlabel='Vorhersage',ylabel='Wahrheit',title='Konfusionsmatrix')
    fpr,tpr,_=roc_curve(truth,prob);axes[1].plot(fpr,tpr);axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set(xlabel='Falsch-positiv-Rate',ylabel='Recall',title=f'ROC · AUC {roc_auc_score(truth,prob):.3f}')
    precision,recall,_=precision_recall_curve(truth,prob);axes[2].plot(recall,precision);axes[2].set(xlabel='Recall',ylabel='Precision',title='Precision–Recall');fig.tight_layout()
    examples=X.reset_index(drop=True).copy();examples['Wahrheit']=y.to_numpy();examples['Vorhersage']=pred;examples['Score']=np.round(prob,3)
    return display_table(report),fig,display_table(examples.head(30)),f'Accuracy {accuracy_score(y,pred):.3f} · Makro-F1 {f1_score(y,pred,average="macro"):.3f}. Wiederholte Entscheidungen anhand dieser Testdaten beeinflussen deren Unabhängigkeit.'

def importance(state,name):
    model=selected(state,name)
    # Importance uses training data, preserving test set for evaluation.
    r=permutation_importance(model,state['Xtr'],state['ytr'],n_repeats=3,random_state=42,scoring='roc_auc',n_jobs=1)
    frame=pd.DataFrame({'Merkmal':state['Xtr'].columns,'Wichtigkeit':r.importances_mean}).sort_values('Wichtigkeit',ascending=False)
    fig,ax=plt.subplots(figsize=(7,4));sns.barplot(data=frame,x='Wichtigkeit',y='Merkmal',color='#4f70b8',ax=ax);fig.tight_layout();return display_table(frame),fig

def predict_file(state,name,path,threshold):
    model=selected(state,name)
    if not path: raise gr.Error('Bitte eine CSV-Datei hochladen.')
    try:
        df=pd.read_csv(path); cols=list(state['Xtr'].columns); missing=[c for c in cols if c not in df]
        if missing: raise gr.Error('Fehlende Merkmale: '+', '.join(missing))
        p=model.predict_proba(df[cols])[:,list(model.classes_).index(state['positive'])]; neg=next(c for c in model.classes_ if c!=state['positive'])
        result=df[cols].copy();result['Vorhersage']=np.where(p>=threshold,state['positive'],neg);result['Score']=np.round(p,3)
        with tempfile.NamedTemporaryFile(suffix='.csv',delete=False) as f: dest=f.name
        result.to_csv(dest,index=False);return display_table(result.head(30)),dest
    except gr.Error: raise
    except Exception as e: raise gr.Error(f'Vorhersage fehlgeschlagen: {e}')

def evaluation_section(state, name_control=None, fixed_name=None):
    gr.Markdown('### Ergebnisse und Entscheidungen prüfen\nSchwelle und Testdaten zeigen Konfusionsmatrix, ROC- und Precision–Recall-Kurven. Wiederholte Modell- oder Schwellenwahl anhand des Tests macht diesen zu Validierungsdaten; in realen Beobachtungen kann der Hintergrund häufiger sein.')
    threshold_local = gr.Slider(0, 1, value=.5, step=.01, label='Schwelle für Signal')
    button = gr.Button('Dieses Modell auf Testdaten auswerten')
    metric_local = gr.Markdown()
    report_local = gr.Dataframe(label='Kennzahlen', interactive=False)
    plot_local = gr.Plot(label='Konfusionsmatrix und Kurven')
    examples_local = gr.Dataframe(label='Beispiele', interactive=False, wrap=True)
    outputs = [report_local, plot_local, examples_local, metric_local]
    if name_control is not None:
        button.click(evaluate, [state, name_control, threshold_local], outputs)
        name_control.change(lambda: (None,None,None,None), outputs=outputs)
    else:
        button.click(lambda st, threshold: evaluate(st, fixed_name, threshold), [state, threshold_local], outputs)
    for control in [state, threshold_local]:
        control.change(lambda: (None,None,None,None), outputs=outputs)

with gr.Blocks(title='KI für die Wissenschaft – Tabellenklassifikation',delete_cache=(3600,86400)) as app:
    gr.Markdown('# KI für die Wissenschaft\n### Tabellarische Klassifikation: Gamma-Signal oder Hintergrund?\nSimulierte MAGIC-Teleskopdaten · Daten verstehen · Modelle vergleichen · Vorhersagen prüfen',elem_id='hero')
    raw=gr.State(None); cleaned=gr.State(None); state=gr.State(None);pg=gr.State(0)
    with gr.Column(elem_classes=['step']):
        gr.Markdown('## 1 · Daten erkunden\nDie Daten enthalten simulierte Teleskopereignisse: `g` steht für Gamma-Signal, `h` für hadronischen Hintergrund. [Hugging Face](https://huggingface.co/datasets/jason1966/abhinand05_magic-gamma-telescope-dataset) · [UCI-Dokumentation und Zitierung](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope). Technisch: `pandas` und `Gradio`.')
        with gr.Row(): upload=gr.File(label='CSV hochladen',file_types=['.csv'],type='filepath'); load=gr.Button('MAGIC-Beispieldaten laden')
        target=gr.Dropdown(label='Zielspalte');features=gr.Dropdown(label='Merkmale',multiselect=True);table=gr.Dataframe(interactive=False);pinfo=gr.Markdown()
        with gr.Row(): prev=gr.Button('← Zurück');nxt=gr.Button('Weiter →')
        check=gr.Button('Datenqualität untersuchen');overview=gr.Dataframe();balance=gr.Dataframe(label='Klassenverteilung');quality=gr.Markdown()
        outputs=[raw,target,features,table,pg,pinfo];load.click(lambda:load_data(),outputs=outputs);upload.change(load_data,upload,outputs);prev.click(lambda d,n:page(d,n,-1),[raw,pg],[table,pg,pinfo]);nxt.click(lambda d,n:page(d,n,1),[raw,pg],[table,pg,pinfo]);check.click(inspect,[raw,target],[overview,balance,quality])
    with gr.Column(elem_classes=['step']):
        gr.Markdown('## 2 · Merkmale erkunden\nUntersuchen Sie die Rohdaten vor der Bereinigung. Welche Messgrößen unterscheiden die Klassen? Korrelation ist keine Kausalität. Technisch: `pandas` und `seaborn`.')
        x=gr.Dropdown(label='X-Achse');y=gr.Dropdown(label='Y-Achse');explorebutton=gr.Button('Merkmale untersuchen');summary=gr.Dataframe();scatter=gr.Plot();corr=gr.Plot()
    with gr.Column(elem_classes=['step']):
        gr.Markdown('## 3 · Daten bereinigen\nFehlende Zielwerte entfernen und Duplikate untersuchen. Fehlende Merkmale werden später innerhalb der Trainingspipeline ersetzt. Technisch: `pandas` und `scikit-learn`.')
        scaling=gr.Radio(['Standardisierung (z-Score)','Normalisierung (Min–Max)','Keine Skalierung'],value='Standardisierung (z-Score)',label='Skalierung numerischer Merkmale',info='Die Parameter werden erst beim Training aus den Trainingsdaten gelernt; die Exploration zeigt weiterhin Originalwerte.')
        dedup=gr.Checkbox(label='Exakte Duplikate entfernen',value=False);cleanbutton=gr.Button('Bereinigen');cleaninfo=gr.Markdown()
        gr.Markdown('Mittelwert und Stichproben-Standardabweichung der bereinigten Originalwerte, vor Imputation und Skalierung. Fehlende Werte werden je Merkmal ausgeschlossen.')
        clean_stats=gr.Dataframe(label='Numerische Merkmale nach Bereinigung',interactive=False)
        gr.Markdown('### Skalierung direkt überprüfen\nDiese Vorschau berechnet die Skalierung auf allen bereinigten Daten, nur zur Veranschaulichung. Für Modelle werden die Skalierungsparameter separat ausschließlich auf den Trainingsdaten gelernt. Nach Imputation werden Mittelwert, Standardabweichung (Division durch N), Minimum und Maximum der skalierten Werte gezeigt.')
        preview_stats=gr.Dataframe(label='Vor und nach der Skalierung – Vorschau',interactive=False)
    cleanbutton.click(clean,[raw,target,features,dedup],[cleaned,cleaninfo,x,y,clean_stats]).then(scaling_preview,[cleaned,target,scaling],preview_stats);explorebutton.click(explore,[raw,target,features,x,y],[summary,scatter,corr])
    def feature_axes(selected):
        selected = selected or []
        return gr.update(choices=selected,value=selected[0] if selected else None), gr.update(choices=selected,value=selected[min(1,len(selected)-1)] if selected else None)
    features.change(feature_axes,features,[x,y])
    scaling.change(scaling_preview,[cleaned,target,scaling],preview_stats)
    for control in [raw,target,features,dedup]: control.change(lambda:None,outputs=preview_stats)
    for control in [raw,target,features,x,y]: control.change(lambda:(None,None,None),outputs=[summary,scatter,corr])
    with gr.Column(elem_classes=['step']):
        gr.Markdown('## 4 · Einfaches Modell und Baseline\nEine feste stratifizierte Aufteilung trennt Training und Test. Vergleichen Sie Mehrheitsklasse und logistische Regression. Technisch: `scikit-learn`.')
        test_fraction=gr.Slider(0.1,0.5,value=0.2,step=0.05,label='Anteil der Testdaten',info='Der Rest wird zum Training verwendet. Die Aufteilung erhält die Klassenanteile.')
        split=gr.Button('Trainings- und Testdaten erstellen');splitinfo=gr.Markdown()
        gr.Markdown('### Skalierung überprüfen\nVergleich auf den Trainingsdaten: Fehlende Werte werden vor der Skalierung imputiert. Standardabweichung mit Division durch N; Standardisierung ergibt bei nicht konstanten Merkmalen ungefähr 0 und 1, Min–Max-Skalierung Minimum 0 und Maximum 1. Testdaten werden mit denselben Trainingsparametern transformiert.')
        scaling_stats=gr.Dataframe(label='Trainingsdaten vor und nach der Skalierung',interactive=False)
        base=gr.Button('Baseline und Regression trainieren');traininfo=gr.Markdown(); gr.Markdown('Die Tabelle zeigt Mittelwerte aus dreifacher Kreuzvalidierung auf den Trainingsdaten. Die Testdaten können darunter separat ausgewertet werden.'); baseline_results=gr.Dataframe(label='Vergleich der einfachen Modelle',interactive=False);model=gr.Dropdown(label='Modell für Auswertung und Vorhersage')
        gr.Markdown('### Baseline-Modelle verstehen\nPermutation Importance misst die Änderung der ROC-AUC auf den Trainingsdaten. Bei der Mehrheitsklasse ist keine Merkmalswirkung zu erwarten; die Werte sind keine kausalen Effekte.')
        baseline_select=gr.Radio(['Mehrheitsklasse','Logistische Regression'],value='Logistische Regression',label='Baseline-Modell')
        baseline_imp=gr.Button('Baseline-Merkmalswichtigkeit berechnen'); baseline_imp_table=gr.Dataframe(label='Merkmalswichtigkeit',interactive=False); baseline_imp_plot=gr.Plot()
        baseline_imp.click(importance,[state,baseline_select],[baseline_imp_table,baseline_imp_plot])
        baseline_select.change(lambda:(None,None),outputs=[baseline_imp_table,baseline_imp_plot])
        evaluation_section(state, name_control=baseline_select)
    with gr.Column(elem_classes=['step']):
        gr.Markdown('## 5 · Komplexeres Modell\nEin Random Forest kombiniert viele Entscheidungsbäume. Vergleichen Sie ihn mit der Regression. Technisch: `scikit-learn`.')
        forest=gr.Button('Random Forest trainieren')
        gr.Markdown('### Random Forest verstehen\nPermutation Importance auf den Trainingsdaten zeigt, welche Merkmale dieses Modell nutzt; sie belegt keine Kausalität.')
        forest_imp=gr.Button('Random-Forest-Merkmalswichtigkeit berechnen'); forest_imp_table=gr.Dataframe(label='Merkmalswichtigkeit',interactive=False); forest_imp_plot=gr.Plot()
        forest_imp.click(lambda s:importance(s,'Random Forest'),state,[forest_imp_table,forest_imp_plot])
        evaluation_section(state, fixed_name='Random Forest')
    with gr.Column(elem_classes=['step']):
        gr.Markdown('## 6 · Einstellungen automatisch suchen\nOptuna vergleicht Einstellungen mittels dreifacher Kreuzvalidierung nur auf Trainingsdaten; optimiert wird ROC-AUC. Technisch: `optuna` und `scikit-learn`.')
        trials=gr.Slider(2,15,value=5,step=1,label='Anzahl Versuche');tune_btn=gr.Button('Suche starten');tune_table=gr.Dataframe(label='Kreuzvalidierung')
        gr.Markdown('### Optimiertes Modell verstehen\nVergleichen Sie die Permutation Importance des durch Optuna gewählten Modells mit dem vorherigen Random Forest.')
        tuned_imp=gr.Button('Optuna-Modell verstehen'); tuned_imp_table=gr.Dataframe(label='Merkmalswichtigkeit',interactive=False); tuned_imp_plot=gr.Plot()
        tuned_imp.click(lambda s:importance(s,'Optimierter Random Forest'),state,[tuned_imp_table,tuned_imp_plot])
        evaluation_section(state, fixed_name='Optimierter Random Forest')
    split.click(split_data,[cleaned,target,scaling,test_fraction],[state,splitinfo,scaling_stats]);base.click(train_baseline_with_evaluation,state,[state,model,traininfo,baseline_results]);forest.click(lambda s:fit_models(s,'forest'),state,[state,model,traininfo]);tune_btn.click(tune,[state,trials],[state,tune_table,model])
    with gr.Column(elem_classes=['step']):
        gr.Markdown('## 7 · AutoML: Modelle vergleichen\nWie in der AutoML-App vergleicht eine budgetierte Zufallssuche verschiedene Modellfamilien mit dreifacher Kreuzvalidierung auf Trainingsdaten. Das beste Modell wird nach Makro-F1 ausgewählt. Technisch: `scikit-learn` und `RandomizedSearchCV`; das ist nicht auto-sklearn.')
        automl_budget=gr.Slider(7,28,value=7,step=1,label='Maximale Anzahl getesteter Konfigurationen',info='Mindestens eine pro Modellfamilie; größere Budgets dauern länger.')
        automl_button=gr.Button('AutoML-Suche starten',variant='primary'); automl_info=gr.Markdown(); automl_table=gr.Dataframe(label='Modellvergleich',interactive=False); automl_plot=gr.Plot(label='Kreuzvalidierung')
        automl_button.click(automl_search,[state,automl_budget],[state,automl_table,automl_plot,automl_info,model])
        gr.Markdown('### Bestes Modell verstehen\nPermutation Importance auf den Trainingsdaten zeigt modellabhängige Zusammenhänge und keine Kausalität.')
        automl_imp=gr.Button('AutoML-Merkmalswichtigkeit berechnen'); automl_imp_table=gr.Dataframe(interactive=False); automl_imp_plot=gr.Plot()
        automl_imp.click(lambda st:importance(st,'AutoML: bestes Modell'),state,[automl_imp_table,automl_imp_plot])
        evaluation_section(state,fixed_name='AutoML: bestes Modell')
    with gr.Column(elem_classes=['step']):
        gr.Markdown('## 8 · Modell anwenden\nLaden Sie eine CSV mit denselben Merkmalen hoch. Scores sind nicht unbedingt kalibrierte Wahrscheinlichkeiten.')
        threshold=gr.Slider(0,1,value=.5,step=.01,label='Schwelle für Signal')
        newdata=gr.File(label='CSV mit neuen Beobachtungen',file_types=['.csv'],type='filepath');predict=gr.Button('Vorhersagen');predictions=gr.Dataframe();download=gr.File(label='Ergebnisse herunterladen');predict.click(predict_file,[state,model,newdata,threshold],[predictions,download])
    for control in [raw,target,features,dedup]:
        control.change(lambda:(None,None,None),outputs=[cleaned,state,model])
        control.change(lambda:(None,None),outputs=[clean_stats,cleaninfo])
    cleaned.change(lambda:(None,None,None),outputs=[state,model,baseline_results])
    for control in [scaling,test_fraction]: control.change(lambda:(None,None,None,None,None,None,None,None,None),outputs=[state,model,baseline_results,baseline_imp_table,baseline_imp_plot,forest_imp_table,forest_imp_plot,tuned_imp_table,tuned_imp_plot])
    for control in [raw,target,features,dedup,cleaned,scaling,test_fraction]: control.change(lambda:(None,None),outputs=[scaling_stats,splitinfo])
    split.click(lambda: None,outputs=baseline_results)
    for control in [state,model,threshold]: control.change(lambda:(None,None),outputs=[predictions,download])
    for control in [raw,target,features,dedup,cleaned]: control.change(lambda:(None,None,None,None,None,None),outputs=[baseline_imp_table,baseline_imp_plot,forest_imp_table,forest_imp_plot,tuned_imp_table,tuned_imp_plot])
    split.click(lambda:(None,None,None,None,None,None),outputs=[baseline_imp_table,baseline_imp_plot,forest_imp_table,forest_imp_plot,tuned_imp_table,tuned_imp_plot])
    base.click(lambda:(None,None),outputs=[baseline_imp_table,baseline_imp_plot])
    forest.click(lambda:(None,None),outputs=[forest_imp_table,forest_imp_plot])
    tune_btn.click(lambda:(None,None),outputs=[tuned_imp_table,tuned_imp_plot])
    for control in [raw,target,features,dedup,cleaned,scaling,test_fraction]: control.change(lambda:(None,None,None,None,None),outputs=[automl_table,automl_plot,automl_info,automl_imp_table,automl_imp_plot])
    split.click(lambda:(None,None,None,None,None),outputs=[automl_table,automl_plot,automl_info,automl_imp_table,automl_imp_plot])
    automl_button.click(lambda:(None,None),outputs=[automl_imp_table,automl_imp_plot])
app.queue(default_concurrency_limit=1)
if __name__=='__main__': app.launch(theme=gr.themes.Base(primary_hue='blue',neutral_hue='slate'),css=CSS)
