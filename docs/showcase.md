# SortWise — content pack for a portfolio page

Everything needed to build a case-study page about this project on a separate website: copy, numbers, assets and the rules for using them. The page **describes** the project and links to the code; it is not a live demo (the app runs locally — see the README).

Source of truth for every number: [results.md](results.md) and [data/results.json](data/results.json). Do not round differently, add numbers, or invent claims.

## Links

| What | URL |
| --- | --- |
| Code (GitHub) | https://github.com/gmorlo/explainable-image-classifier |
| Full results write-up | https://github.com/gmorlo/explainable-image-classifier/blob/main/docs/results.md |
| Versioned model (Hugging Face) | https://huggingface.co/gmorlo/sortwise-efficientnet-b0 |

## Assets

Copy these files into the website's own static folder (do not hotlink `raw.githubusercontent.com` for the video — it is not served as a video type). All are pinned to the `v1.0` tag so they do not change under the page.

Base URL: `https://raw.githubusercontent.com/gmorlo/explainable-image-classifier/v1.0/docs/`

| File | Use | Size |
| --- | --- | --- |
| `demo.mp4` | Hero video, 22 s, 1920×1080, with music. Start muted; show controls. | 2.1 MB |
| `img/demo_poster.jpg` | Poster / thumbnail for the video and for social cards | 0.2 MB |
| `img/demo.gif` | Fallback where video is not allowed | 3.8 MB |
| `img/app_error.png` | App screenshot: cardboard predicted as paper (96%) with heatmap | 0.3 MB |
| `img/gradcam_correct.png` | Grad-CAM gallery, one correct test prediction per class | 0.6 MB |
| `img/gradcam_errors.png` | Grad-CAM gallery of test errors, most common confusions first | 1.2 MB |
| `img/chart_models.png` | Backbone comparison (head only vs fine-tune) | 0.1 MB |
| `img/chart_seeds.png` | Seed spread for the top-2 learning rates | 0.1 MB |
| `img/chart_per_class.png` | Per-class test F1 | 0.1 MB |
| `img/chart_training.png` | Validation macro-F1 per epoch, early stopping | 0.1 MB |
| `img/test_confusion_matrix.png` | Test confusion matrix | 0.1 MB |
| `data/results.json` | All chart data, if the site prefers to draw charts natively in its own style | 8 KB |

Charts are PNGs on a light surface. If the site has a dark theme, either place them on a light card or redraw them from `data/results.json`.

## Page copy

### Hero
**SortWise — an explainable waste classifier**
Upload a photo of one waste item and get the three most likely categories plus a Grad-CAM heatmap of what the model looked at. Built as an engineering project: audited data, reproducible training, a test set used once.

Stat tiles:
- **0.893** test macro-F1
- **0.910** test accuracy
- **379** held-out test images, used once
- **~44 ms** per image incl. heatmap, desktop CPU

### The problem
Sorting waste starts with recognising the material. TrashNet is a public dataset of 2,527 photos in six classes (cardboard, glass, metal, paper, plastic, other). Training a CNN on it is easy; knowing whether the score is real, reproducible and explainable is the actual work.

### What I did (process)
1. **Audited the data.** SHA-256 checksums found 3 pairs of byte-identical images with different labels (e.g. the same file as both glass and plastic). The copies leaked between train and validation. All 6 were removed; 2,521 images remain, split 70/15/15, stratified.
2. **Made training reproducible.** The same config scored 0.896 one run and 0.928 the next. The seed was set after the model was built and cuDNN chose kernels non-deterministically. After the fix, reruns are identical.
3. **Chose the model by measurement.** ResNet18, EfficientNet-B0 and MobileNetV3-Small, each trained head-only and with the last stage fine-tuned. Fine-tuning added +0.06–0.13 macro-F1. EfficientNet-B0 won: best validation score (0.914) at about a third of ResNet18's weight size (16 MB vs 45 MB). *(chart_models.png)*
4. **Tuned honestly.** A 12-run grid, then the top two learning rates rerun on three seeds. The gap between them (0.008) was smaller than the seed-to-seed spread, so I picked the more stable one (0.919 ± 0.015) instead of the best single run. *(chart_seeds.png)*
5. **Evaluated once.** Macro-F1 as the main metric because the "other" class is 3–4× smaller. The test set was touched once, at the end: 0.893 macro-F1 vs 0.925 on validation — the drop is reported, not hidden. *(chart_per_class.png)*
6. **Shipped it.** A Gradio app with top-3 classes and a heatmap, the model published to the Hugging Face Hub under an immutable version tag (rollback = one environment variable), and CI tests on every push.

### What Grad-CAM showed
On correct predictions the model looks at the object — bottle necks, cans — not the white background. The errors are explainable:
- **Cardboard → paper** (most common, 5×): flat cardboard without visible corrugation; the heatmap sits on edges and folds that look like a sheet of paper.
- **Plastic → glass** (5×): clear or green bottles seen from the neck; the round opening is typical of glass bottles.
- **Metal → glass**: shiny, reflective cans.

Many mistakes come with 84–100% confidence, so a confidence threshold alone would not catch them. That is why the app shows the top three classes instead of one answer. *(app_error.png, gradcam_errors.png)*

### Limitations (keep this section — it is part of the pitch)
- TrashNet photos are single items on a plain light background; other backgrounds are not measured.
- Six known classes only: anything else still gets one of them.
- Softmax scores are model confidence, not calibrated probabilities.
- Grad-CAM shows what raised the class score — not an object mask or proof of correct reasoning.

### Stack
PyTorch · torchvision · EfficientNet-B0 · Grad-CAM (own hook-based implementation) · scikit-learn · Gradio · Hugging Face Hub · GitHub Actions · pytest

### Next
Evaluation on my own photos with other backgrounds; analysis history with user corrections; an in-browser demo (ONNX).

## Rules for the page
- English; numbers exactly as above.
- Credit the data: "TrashNet by Gary Thung and Mindy Yang" with a link to https://github.com/garythung/trashnet.
- No live-demo claims and no "try it" button — the app runs locally.
- Do not show personal email addresses. Contact links come from the site itself.

## Prompt for the website's Claude session

```
Build a case-study page for my project SortWise on my portfolio site, as its own subpage (e.g. /projects/sortwise), in the site's existing style and stack.
Everything you need is in https://github.com/gmorlo/explainable-image-classifier/blob/v1.0/docs/showcase.md — read it first and follow its "Rules for the page".
Copy the listed assets from the v1.0 tag into the site's static folder; don't hotlink the video.
Use only the copy and numbers from that file. Add a card for the project on the projects index that links to the new page.
Show me the page locally before deploying anything.
```
