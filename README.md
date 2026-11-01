# ML for Science

Interactive Gradio labs for learning machine learning. Apps share the blue/slate
theme in `styles.css`.

| App | Topic |
| --- | --- |
| `app.py` | Tabular classification and regression |
| `automl_app.py` | Automated model selection |
| `kumo_app.py` | Tabular foundation models |
| `sam_app.py` | Image segmentation with SAM |
| `zeroshot_app.py` | Multilingual zero-shot text classification |
| `transfer_app.py` | Transfer learning for leaf image classification |

The transfer learning lab reuses the frozen backbone of Hugging Face's
[MobileNetV2 image classifier](https://huggingface.co/google/mobilenet_v2_1.0_224)
and trains a new three-class head on the small
[Beans leaf dataset](https://huggingface.co/datasets/AI-Lab-Makerere/beans).
It compares pretrained features against resized pixels using the same images,
shows learning curves and validation errors, supports image uploads for inference,
and exports the trained head and experiment results.

Use Python 3.12 or 3.13 in a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-transfer.txt
python transfer_app.py
```

The transfer lab uses Gradio 6.31+, Transformers 5.x and Hugging Face Hub 1.x.
For a dedicated Hugging Face Space, use `requirements-transfer.txt` as the Space's
`requirements.txt` and set its app file to `transfer_app.py`. Transformers 4.x
requires Hub below 1.0 and conflicts with Gradio 6.31's Hub requirement.

Open the local Gradio URL printed in the terminal (usually http://127.0.0.1:7860).
No Hugging Face token or GPU is required. Initial downloads are approximately
177 MB (14.2 MB weights, 144 MB training data, 18.5 MB validation data), plus
installed dependencies. Files use the standard Hugging Face cache; features are
cached in memory. A smaller training budget reduces compute, not download size.

The default experiment uses 20 images per class (60 total). Training subsets are
balanced, seeded and nested as the budget increases. Both classifiers use Adam,
training-only feature normalization and the same epoch/learning-rate settings.
Only the new head is trained; this lesson demonstrates linear probing rather than
updating backbone weights. CPU is supported; CUDA is used for feature extraction
when available. Each browser session retains its own trained head.

All 133 official validation images are used for exploration. The 128 official
test images are not loaded or evaluated. Repeated validation comparisons are
model selection, not an unbiased final performance estimate. Epoch zero shows
a randomly initialized leaf head on each representation; the original ImageNet
labels are not bean disease labels.

Downloads include `report.json` (settings, revisions, sample indices and metrics),
learning curves, validation predictions, and `leaf_head.npz` (weights, bias and
training normalization). Inference instructions accompany the exported head.
See [source notes](sources/transfer_learning.md) for model/data provenance and terms.

Run the offline training, export and UI construction checks with:

```bash
python -m unittest discover -s tests -v
```
