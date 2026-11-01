"""SAM segmentation teaching app: python sam_app.py."""
from pathlib import Path
from threading import Lock
import json
import colorsys
import tempfile
import time
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import gradio as gr

MODEL_ID = 'facebook/sam-vit-base'
LOCK = Lock()
BACKEND = None

def backend():
    global BACKEND
    if BACKEND is None:
        try:
            import torch
            from transformers import SamModel, SamProcessor, pipeline
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SamModel.from_pretrained(MODEL_ID).to(device).eval()
            processor = SamProcessor.from_pretrained(MODEL_ID)
            generator = pipeline('mask-generation', model=model, image_processor=processor.image_processor, device=device)
            BACKEND = torch, model, processor, generator, device
        except Exception as exc:
            raise gr.Error(f'Cannot load SAM. Install requirements-sam.txt and check Hugging Face download access: {exc}') from exc
    return BACKEND

def prepare(image):
    if image is None:
        return None, [], None, None, None, 'Upload an image or capture a webcam snapshot.', None, None
    image = ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail((1024, 1024))
    return image, [], image, None, None, f'Processing dimensions: {image.width} × {image.height}. Add foreground points or find masks.', None, None

def marked(image, points):
    if image is None:
        return None
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for x, y, label in points:
        draw.ellipse((x-5,y-5,x+5,y+5), fill='lime' if label else 'red', outline='white', width=2)
    return result

def click(image, points, label, evt: gr.SelectData):
    if image is None:
        raise gr.Error('Upload an image or capture a webcam snapshot first.')
    x, y = map(int, evt.index)
    if not (0 <= x < image.width and 0 <= y < image.height):
        raise gr.Error('Click inside the image.')
    points = [*(points or []), (x,y,int(label == 'Foreground'))]
    return points, marked(image, points), None, None, None, None

def undo(image, points, clear=False):
    points = [] if clear else (points or [])[:-1]
    return points, marked(image, points), None, None, None, None

def mask_overlay(image, masks, opacity=0.85):
    """Paint large regions first so smaller regions remain visible in overlaps."""
    source = np.asarray(image.convert('RGB'))
    result = source.copy()
    for index in sorted(range(len(masks)), key=lambda i: np.count_nonzero(masks[i]), reverse=True):
        mask = np.asarray(masks[index], dtype=bool)
        color = np.asarray(colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.85, 1.0)) * 255
        result[mask] = np.rint(source[mask] * (1-opacity) + color * opacity).astype('uint8')
    return Image.fromarray(result)

def run(image, points, automatic, grid, quality, stability):
    if image is None:
        raise gr.Error('Upload an image or capture a webcam snapshot first.')
    if not automatic and not any(p[2] for p in points):
        raise gr.Error('Add at least one foreground point.')
    start = time.perf_counter()
    try:
        with LOCK:
            torch, model, processor, generator, device = backend()
            with torch.inference_mode():
                if automatic:
                    output = generator(image, points_per_crop=int(grid), points_per_batch=32 if device == 'cuda' else 4, pred_iou_thresh=quality, stability_score_thresh=stability, crops_n_layers=0)
                    masks, scores = output['masks'], output['scores']
                else:
                    inputs = processor(image, input_points=[[[x,y] for x,y,_ in points]], input_labels=[[[label for _,_,label in points]][0]], return_tensors='pt').to(device)
                    output = model(**inputs)
                    masks = processor.image_processor.post_process_masks(output.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())[0][0].numpy()
                    scores = output.iou_scores[0,0].cpu().numpy()
        mask_count = len(masks)
        order = np.argsort(np.asarray(scores, dtype=float))[::-1][:100]
        records = []
        gallery = []
        directory = Path(tempfile.mkdtemp(prefix='sam_results_'))
        retained_masks = []
        for i, index in enumerate(order):
            mask = np.asarray(masks[index], dtype=bool)
            binary = Image.fromarray(mask.astype('uint8')*255)
            binary.save(directory / f'mask_{i+1:03}.png')
            retained_masks.append(mask)
            gallery.append((binary, f'Mask {i+1}: estimated quality {float(scores[index]):.3f}'))
            records.append({'mask':i+1,'estimated_quality':float(scores[index]),'area_pixels':int(mask.sum())})
        overlay = mask_overlay(image, retained_masks)
        overlay.save(directory/'overlay.png')
        report = dict(model=MODEL_ID, mode='automatic' if automatic else 'points', dimensions=image.size, overlay_opacity=0.85, mask_count=mask_count, retained_mask_count=len(records), points=points, grid=int(grid), quality=quality, stability=stability, seconds=time.perf_counter()-start, masks=records)
        (directory/'report.json').write_text(json.dumps(report,indent=2))
        archive = directory/'results.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
            for path in directory.iterdir():
                if path != archive:
                    bundle.write(path,path.name)
        return gallery, str(archive), f'{mask_count} masks found · {len(records)} shown/exported · {report["seconds"]:.2f} seconds · {device}. Masks may overlap; scores are estimates, not measured accuracy.', overlay, mask_count
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f'Segmentation failed. Try fewer sampling points or check available memory: {exc}') from exc

with gr.Blocks(title='SAM Segmentation Science Lab', delete_cache=(3600,86400)) as demo:
    gr.Markdown('# SAM Segmentation Science Lab\nExplore a pretrained segmentation foundation model.', elem_id='lab-header')
    gr.Markdown('Upload or capture a snapshot → mark foreground/background → segment, or discover masks automatically. Weights stay fixed. Masks have no class labels.', elem_classes=['lab-note'])
    original, points = gr.State(None), gr.State([])
    with gr.Row():
        with gr.Column(elem_id='lab-controls'):
            image = gr.Image(type='pil', sources=['upload', 'webcam'], streaming=False, label='Upload an image or take a webcam snapshot; click to add points')
            label = gr.Radio(['Foreground','Background'],value='Foreground',label='Point type')
            back = gr.Button('Undo last point')
            clear = gr.Button('Clear points')
            segment = gr.Button('Segment object',variant='primary')
            grid = gr.Radio([8,16,32],value=8,label='Automatic sampling points per side')
            quality = gr.Slider(0,1,value=.88,label='Minimum predicted quality')
            stability = gr.Slider(0,1,value=.95,label='Minimum stability')
            automatic = gr.Button('Find masks')
        with gr.Column(elem_id='lab-results'):
            status = gr.Markdown('Upload an image or take a webcam snapshot to begin. Processing is limited to a 1,024-pixel side; exports use processing dimensions.')
            mask_count_view = gr.Number(value=None, label='Masks found', precision=0, interactive=False, info='Mask proposals, not distinct objects. Up to 100 are shown and exported.')
            with gr.Tabs():
                with gr.Tab('Color overlay'):
                    overlay_view = gr.Image(label='Masks highlighted at 85% opacity', type='pil', interactive=False)
                    gr.Markdown('Each mask has a distinct color. Smaller masks appear on top in overlapping regions; unmasked pixels retain their original colors.')
                with gr.Tab('Individual masks'):
                    gallery = gr.Gallery(label='Masks ranked by estimated quality',columns=3)
            download = gr.File(label='Download masks, overlay and report')
    # User input includes uploads, webcam snapshots and removal. Backend marker
    # updates do not fire input, so adding a point never resets the image state.
    image.input(prepare,[image],[original,points,image,gallery,download,status,overlay_view,mask_count_view])
    image.select(click,[original,points,label],[points,image,gallery,download,overlay_view,mask_count_view])
    back.click(undo,[original,points],[points,image,gallery,download,overlay_view,mask_count_view])
    clear.click(lambda a,b:undo(a,b,True),[original,points],[points,image,gallery,download,overlay_view,mask_count_view])
    for control in [grid,quality,stability]:
        control.change(lambda:(None,None,None,None),outputs=[gallery,download,overlay_view,mask_count_view])
    segment.click(lambda *args:run(args[0],args[1],False,*args[2:]),[original,points,grid,quality,stability],[gallery,download,status,overlay_view,mask_count_view],concurrency_id='sam',concurrency_limit=1)
    automatic.click(lambda *args:run(args[0],args[1],True,*args[2:]),[original,points,grid,quality,stability],[gallery,download,status,overlay_view,mask_count_view],concurrency_id='sam',concurrency_limit=1)
    gr.Markdown('### Try it\n1. Segment an object with one foreground point.\n2. Add background points to exclude nearby regions.\n3. Compare automatic sampling densities and thresholds.\n\n[SAM Base weights and model card](https://huggingface.co/facebook/sam-vit-base) · Apache 2.0. First use downloads weights; CPU inference may be slow.')
demo.queue(max_size=20,default_concurrency_limit=1)
if __name__ == '__main__':
    demo.launch(theme=gr.themes.Base(primary_hue='blue',neutral_hue='slate'),css_paths=[Path(__file__).with_name('styles.css')])
