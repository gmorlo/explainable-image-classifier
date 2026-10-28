# explainable-image-classifier

## Context for AI

This document is the **master plan for the project** and the reference point for AI assisting with implementation. The features described are planned; their presence here does not mean they already exist in the code. Check the actual contents of the repository before making changes.

The project can cover nearly the whole ML Engineer workflow. The most valuable thing it can show is that you can **justify decisions, measure quality, deploy a model and maintain it after deployment**. Combined with requirements analysis, architecture, databases, testing and project management, it gives material for technical interviews that goes beyond an accuracy score.

Working name of the application: **SortWise**. Repository name: `explainable-image-classifier`.

---

## Project goal

Build a web application that classifies a single piece of waste in a photo into one of six categories and shows a Grad-CAM visualization. The user uploads a photo, sees the result, checks the regions that influenced the prediction and can correct a wrong category.

**Target portfolio outcome:**  
"I built an image classification application with reproducible training, quality evaluation, automated deployment, monitoring and a controlled model update."

The priority is a finished, measurable product delivered in three releases (V1 → V2 → V3), not building the whole infrastructure at once.

---

## Business use case

Support a user or operator in the initial recognition of waste material. The system suggests a category; the user can verify it. Potential use: a helper step in a sorting process.

The MVP recognizes the waste category. It does not choose a bin based on local rules, assess contamination or control a physical sorting facility. Effectiveness in a real process requires separate validation.

---

## Data

We use the ready-made **TrashNet** dataset (Gary Thung, Mindy Yang).

- Source: https://github.com/garythung/trashnet
- Original dataset: 2527 images, mostly single items on a white background.
- Collecting our own training set is not a precondition for starting.
- The dataset is not added to Git. Before training we verify the version, labels, counts, duplicates and terms of use.

| Source label | UI label | Image count |
| --- | --- | --- |
| `cardboard` | Cardboard | 403 |
| `glass` | Glass | 501 |
| `metal` | Metal | 410 |
| `paper` | Paper | 594 |
| `plastic` | Plastic | 482 |
| `trash` | Other waste | 137 |

The `trash` class is a specific class from the dataset, not a catch-all bin for "unknown objects".

---

## ML stack — technical decision

| Element | Decision |
| --- | --- |
| Framework | **PyTorch** (current stable version) + **torchvision** |
| Acceleration | **CUDA** — training and local evaluation on an NVIDIA GPU |
| Inference in the demo | GPU preferred if available; CPU fallback in the container / Cloud Run |
| Models to compare | At least **two** backbones (see experiments) |
| Interpretation | Grad-CAM |

Training is designed for CUDA from the start: device detection, `pin_memory`, moving batches to the GPU, and recording training/evaluation timing metrics together with hardware information. The code must also run on CPU (CI, Cloud Run without a GPU), but local experiments assume an NVIDIA card.

---

## Experiments: models and hyperparameters

Comparing solutions and justifying the choice is one of the five key elements of the portfolio. "Just take ResNet18" is not enough.

### Model comparison (architecture matrix)

We compare **at least two** models using the same data split, the same validation preprocessing and the same evaluation procedure:

| Model | Role | Fine-tuning levels to check |
| --- | --- | --- |
| **ResNet18** | Baseline — simple, fast, well known | (A) frozen backbone + trained head; (B) fine-tuning part/the end of the network |
| **EfficientNet-B0** or **MobileNetV3-Small** | Alternative — different quality/size/time balance | The same A/B variants |

Result: a table with macro-F1, accuracy, training time, inference time (GPU and CPU), weight size and a short justification for the production model choice.

### Hyperparameter matrix

For the chosen baseline (ResNet18 first, variant A or B) we run a **limited grid** — not a full grid of everything at once. The goal is to show the methodology, not to exhaust the GPU.

Proposed starting range (to be refined after the first run):

| Hyperparameter | Values to check |
| --- | --- |
| Learning rate | `1e-3`, `3e-4`, `1e-4` |
| Batch size | `16`, `32` (adjust to VRAM) |
| Optimizer | `Adam` vs `SGD + momentum` (one axis at a time relative to LR) |
| Weight decay | `0`, `1e-4` |
| Epochs / early stopping | fixed max + early stopping on validation macro-F1 |
| Augmentations | minimal vs extended set (train only) |

Rules:

1. First compare model variants (A/B × 2 architectures) with **fixed** hyperparameters.
2. Then run grid/random search over hyperparameters **only for the finalist** (or top-2).
3. Log all runs in an experiment tracker (e.g. MLflow): parameters, metrics, data version, commit, weights.
4. Justify the final choice with measurements (quality vs time vs size), not intuition.

---

## MVP scope (product V1)

1. Upload a single JPEG/PNG image.
2. File validation: format, decoding, size and pixel limits.
3. Classification of one dominant item into one of six classes.
4. Display of the category and the scores for all classes.
5. Grad-CAM overlaid on the image.
6. Saving the analysis together with the model version.
7. Confirming or correcting the category.
8. Analysis history in the interface.

V1 does not implement: multi-object detection, segmentation, video, a mobile app, login, automatic retraining on corrections. User feedback is saved but does not change the weights automatically.

---

## Full lifecycle — 16 stages

This approach follows the MLOps cycle (data → training → validation → deployment → monitoring → retraining). We implement it gradually across releases V1–V3.

| Stage | What you do in the project | Result |
| --- | --- | --- |
| **1. Problem and requirements** | User, scenario, 6 classes, constraints, response time | Specification, use cases, acceptance criteria |
| **2. Architecture design** | Training, inference, API, frontend, storage; justification of choices | Diagram, ERD, short ADRs |
| **3. Data acquisition and versioning** | Download TrashNet, version, provenance | Script + manifest with checksums |
| **4. EDA** | Counts, resolutions, corrupt files, duplicates, background | Notebook/report with conclusions |
| **5. Data preparation** | Fixed train/val/test split, preprocessing, augmentations on train only | Pipeline without leakage |
| **6. Baseline model and experiments** | Architecture comparison + hyperparameter matrix | Experiment table and model choice |
| **7. Experiment tracking** | Parameters, metrics, data, commit, weights | History in MLflow (or equivalent) |
| **8. Evaluation and error analysis** | Macro-F1, per-class, confusion matrix; paper–cardboard; own photos | Quality report with examples |
| **9. Explainability** | Grad-CAM: object vs background | Gallery of correct and incorrect predictions |
| **10. Model packaging** | Weights + preprocessing + class mapping = versioned package | Reproducible artifact for deployment |
| **11. Serving** | FastAPI, validation, errors, load at startup | API + docs + latency measurements |
| **12. Application and feedback** | Upload, result, history, category correction | Demo with the original prediction and a separate correction |
| **13. Testing** | Preprocessing, labels, API–model–DB, user flow | Automated tests + performance report |
| **14. CI/CD and deployment** | Lint/tests, Docker image, version deploy | Repeatable release + rollback |
| **15. Monitoring** | Errors, latency, model version, prediction distribution, verified corrections | Dashboard and alert rules |
| **16. Retraining** | Label verification → new data version → candidate vs production | Controlled model update |

Source of the MLOps concept: [Google Cloud — MLOps](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning).

---

## Five elements with the highest portfolio value

1. **Experiment reproducibility** — data, split, code, configuration, environment; a second person can repeat a run (allowing for GPU nondeterminism).
2. **Comparing solutions and justification** — e.g. fine-tuning improved macro-F1 but lengthened training; the choice is documented with measurements. Tracker: [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking).
3. **Quality gate before deployment** — a new model ships only after passing thresholds on quality, classes, API compatibility and latency; the test set is used only for the final evaluation.
4. **Rollback** — restoring the previous package (weights + preprocessing + class mapping).
5. **Closing the feedback loop** — correction → verification → new data version → candidate → evaluation → only then production.

---

## Monitoring — three separate layers

| Type | Example measurements | Conclusion |
| --- | --- | --- |
| **Service operation** | Errors, p95 latency, memory | Is the application running properly? |
| **Input data change** | Different backgrounds, brightness, feature distribution | Are photos arriving that differ from training? |
| **Prediction quality** | Macro-F1 on verified labels | Is the model really classifying worse? |

A data change or a drop in softmax scores **does not prove** a drop in quality. Measuring quality requires correct labels; voluntary user corrections alone may be unrepresentative.

---

## Releases: V1 / V2 / V3

| Release | Scope | Goal |
| --- | --- | --- |
| **V1 — working product** | Data, EDA, split, baseline + comparison of 2 models / hyperparameters (within reason), evaluation, Grad-CAM, API, simple UI, history and feedback | A working demo with measurable quality |
| **V2 — engineering workshop** | MLflow, artifact versioning, tests, Docker, CI/CD, deployment, rollback, quality gate | Repeatable releases and safe changes |
| **V3 — full ML loop** | Monitoring (3 layers), verified feedback, retraining, candidate vs production comparison | Controlled model update |

**Possible extensions (described, not required in V1–V3):** Kubernetes, feature store, distributed training, extensive orchestration, A/B tests. For a single CNN and a small dataset they do not yet bring a proportional benefit.

---

## Model and interpreting results

- Adapt the output layer to 6 classes.
- Save the index → label mapping together with the weights.
- The same deterministic preprocessing in evaluation and the API; augmentation on train only.
- Present softmax as a model score — without calibration, do not call it a verified probability.
- Grad-CAM for the chosen class and convolutional layer; do not disable autograd along the whole heatmap path.
- The model may classify a non-waste item into one of the known classes; a low-score threshold can suggest manual review, but it does not solve open-set recognition.

Grad-CAM shows the regions that influence the decision — it is not an object mask or proof of "correct reasoning".

---

## Quality evaluation

- Stratified split, e.g. **70/15/15**; save the image IDs in each subset.
- Duplicates / related images: remove them or keep them in the same fold — no leakage.
- Hyperparameters and thresholds on validation only; **test set only for the final evaluation**.
- Report: accuracy, **macro-F1**, per-class precision/recall, confusion matrix.
- Examples of correct and incorrect predictions + heatmaps.
- A separate set of **own photos** (different backgrounds) — reported separately from the TrashNet test.
- Measure prediction and Grad-CAM time on the described hardware (GPU/CPU).

We do not assume a target performance in advance. Conclusions come from measurements — especially for the small `trash` class and for photos outside the training conditions.

---

## Proposed architecture

| Element | Responsibility | Tools |
| --- | --- | --- |
| ML module | Data, training, evaluation, prediction, Grad-CAM | PyTorch + CUDA, torchvision, MLflow |
| Backend | Validation, analyses, history, feedback | FastAPI |
| Frontend | Upload, result, heatmap, history | To be decided |
| Database | Analysis metadata and corrections | PostgreSQL |
| Environment | Local run | Docker / Compose |
| Deployment | Demonstration | Eventually GCP Cloud Run |

Flow: UI → API → validation → model + Grad-CAM → save → UI. A user correction is a separate record; it does not change the weights automatically.

To start: a single backend, in-process inference, model loaded at startup. Kubernetes / microservices / queues are outside the MVP.

Before Cloud Run: persistent storage (e.g. GCS) + an external database. The container's local disk is not a history store. Image retention and deletion is an explicit decision to make before publishing.

---

## Proposed API contract

A proposal — do not assume the endpoints already exist.

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Availability |
| `POST /analyses` | Analyze an image and save |
| `GET /analyses` | History (pagination) |
| `GET /analyses/{id}` | Single analysis |
| `PUT /analyses/{id}/feedback` | Confirmed / corrected category |

Result: id, date, class, scores for all classes, model version, time, how to fetch the visualization. Feedback is kept separate from the original prediction.

---

## Implementation order (start → V1)

1. **Data:** download, QC, manifest, save the split.
2. **EDA:** notebook with conclusions.
3. **Experiments:** 2 models × A/B variants → hyperparameter grid for the finalist → choice with justification.
4. **Prediction + Grad-CAM** outside the web.
5. **API** with validation and model load at startup.
6. **Interface** + history + feedback.
7. **Documentation** of how to run, limitations and actual measurements.

Each stage ends with a verifiable result. Do not start with elaborate infrastructure (that is V2+).

---

## V1 completion criteria

- Analysis of a valid image: category, class scores, Grad-CAM.
- Invalid file → clear error, no crash.
- Analysis in history; feedback without overwriting the original prediction.
- TrashNet test report + a separate evaluation on own photos.
- Model / hyperparameter comparison table with justification of the choice.
- Weights, label mapping and preprocessing versioned and reproducible.
- Documentation describes the actual run and measurements — no invented results.

---

## Working rules for AI

- Before implementing, check the files, dependencies and repo state. Do not assume code from the plan exists.
- Work on the nearest agreed stage; do not build V2/V3 "along the way" in V1.
- Keep training separate from the API serving code.
- Keep labels and preprocessing consistent between training and inference.
- Do not commit secrets, the dataset or large weights. Document how to obtain them.
- Do not use example results as real predictions.
- Adequate tests: class mapping, API contract, file validation, model–application integration.
- Update the user-facing README after each stage: what works, how to run it, limitations.
- Do not add install/run commands until the corresponding files exist and have been checked.
- Assume local training on **CUDA**; CI and deployment without a GPU must have a CPU path.

---

## Open decisions

- ~~Frontend framework.~~ **Decision (2026-10-28):** demo in **Gradio** (`app.py`), run **locally**; weights in a versioned model repo on the HF Hub. Hosting postponed: Gradio Spaces on the free CPU tier require HF PRO, and the budget should stay close to zero. Presentation via screenshots / a short video. CI = tests in GitHub Actions. FastAPI + Postgres + Cloud Run deferred (history and feedback).
- ~~The exact second model.~~ EfficientNet-B0 chosen — see `docs/results.md`.
- ~~Final scope of the hyperparameter grid.~~ See `docs/results.md`.
- Storage and retention of images / visualizations.
- Threshold for suggesting manual verification (after validation).
- History access and the cost of a public deployment.

**For now:** a portfolio README (engineering process, not "ML on a ready-made dataset") + screenshots / video of the local demo.
