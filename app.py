"""SortWise Gradio demo: upload a photo of one waste item, get top-3 classes and a Grad-CAM heatmap."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sortwise.config import CLASS_LABELS_PL, RUNS_DIR
from sortwise.inference import load_predictor
from sortwise.training.trainer import get_device

RUN_DIR = Path(os.environ.get("SORTWISE_RUN_DIR", RUNS_DIR / "final" / "efficientnet_b0"))
predictor = load_predictor(RUN_DIR, get_device())


def analyze(image):
    if image is None:
        raise gr.Error("Dodaj zdjęcie (JPG lub PNG).")
    scores, heatmap = predictor.predict(image)
    return {CLASS_LABELS_PL[name]: score for name, score in scores.items()}, heatmap


with gr.Blocks(title="SortWise") as demo:
    gr.Markdown(
        "# SortWise\n"
        "Klasyfikacja jednego odpadu na zdjęciu do 6 kategorii (TrashNet) z mapą Grad-CAM."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Zdjęcie", sources=["upload", "clipboard"])
            button = gr.Button("Analizuj", variant="primary")
        with gr.Column():
            label = gr.Label(num_top_classes=3, label="Top-3 klasy (wynik softmax)")
            heatmap = gr.Image(label="Grad-CAM dla klasy top-1", interactive=False)
    gr.Markdown(
        "Wynik softmax to pewność modelu, nie skalibrowane prawdopodobieństwo — model bywa pewny "
        "także przy błędach, dlatego pokazujemy top-3. Heatmapa wskazuje obszary, które najbardziej "
        "podniosły wynik klasy top-1; to nie jest maska obiektu. Model zna tylko 6 klas i był uczony "
        "na zdjęciach przedmiotów na jasnym tle."
    )
    button.click(analyze, inputs=image, outputs=[label, heatmap])
    image.upload(analyze, inputs=image, outputs=[label, heatmap])

if __name__ == "__main__":
    demo.launch(max_file_size="10mb")
