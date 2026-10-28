"""SortWise Gradio demo: upload a photo of one waste item, get top-3 classes and a Grad-CAM heatmap."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sortwise.config import CLASS_LABELS, RUNS_DIR
from sortwise.inference import load_predictor
from sortwise.training.trainer import get_device

RUN_DIR = Path(os.environ.get("SORTWISE_RUN_DIR", RUNS_DIR / "final" / "efficientnet_b0"))
MODEL_REPO = os.environ.get("SORTWISE_MODEL_REPO", "gmorlo/sortwise-efficientnet-b0")
MODEL_VERSION = os.environ.get("SORTWISE_MODEL_VERSION", "v1")  # rollback = set an older tag

if (RUN_DIR / "best.pt").exists():
    MODEL_LABEL = f"local run `{RUN_DIR.name}`"
else:  # e.g. on HF Spaces, where weights are not in Git
    from huggingface_hub import snapshot_download

    RUN_DIR = Path(snapshot_download(MODEL_REPO, revision=MODEL_VERSION))
    MODEL_LABEL = f"`{MODEL_REPO}@{MODEL_VERSION}`"
predictor = load_predictor(RUN_DIR, get_device())


def analyze(image):
    if image is None:
        raise gr.Error("Add a photo (JPG or PNG).")
    scores, heatmap = predictor.predict(image)
    return {CLASS_LABELS[name]: score for name, score in scores.items()}, heatmap


with gr.Blocks(title="SortWise") as demo:
    gr.Markdown(
        "# SortWise\n"
        "Classifies one waste item in a photo into 6 TrashNet categories and shows a Grad-CAM heatmap.  \n"
        f"Model: {MODEL_LABEL}"
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Photo", sources=["upload", "clipboard"])
            button = gr.Button("Analyze", variant="primary")
        with gr.Column():
            label = gr.Label(num_top_classes=3, label="Top-3 classes (softmax score)")
            heatmap = gr.Image(label="Grad-CAM for the top-1 class", interactive=False)
    gr.Markdown(
        "The softmax score is the model's confidence, not a calibrated probability. The model can be "
        "confident when it is wrong, which is why the top 3 are shown. The heatmap marks the regions that "
        "raised the top-1 score most; it is not an object mask. The model knows only 6 classes and was "
        "trained on single items on a plain light background."
    )
    button.click(analyze, inputs=image, outputs=[label, heatmap])
    image.upload(analyze, inputs=image, outputs=[label, heatmap])

if __name__ == "__main__":
    demo.launch(max_file_size="10mb")
