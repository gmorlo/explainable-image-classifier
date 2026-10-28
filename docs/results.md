# Wyniki i wnioski

Stan: 2026-10-28. Wszystkie treningi na NVIDIA RTX 3070 (CUDA), PyTorch 2.11.
Dane: TrashNet `dataset-resized`, 2521 zdjęć po usunięciu duplikatów, split 70/15/15 (seed 42, stratyfikowany).
Główna miara: **macro-F1** — klasa `trash` (137 zdjęć) jest 3–4× mniejsza od pozostałych, accuracy by to ukryło.

## 1. Problemy znalezione w danych i kodzie

| Problem | Skutek | Naprawa |
| --- | --- | --- |
| 3 pary identycznych plików (ten sam SHA-256) z **różnymi etykietami**: `glass115`=`metal91`, `glass176`=`plastic152`, `glass389`=`plastic332` | Błędna etykieta + przeciek: kopie trafiały do train i val | `drop_duplicates` w `split.py` usuwa wszystkie 6 (prawdziwa etykieta nieznana); lista w `splits.json → excluded` |
| Seed ustawiany dopiero po zbudowaniu modelu, cuDNN autotuning | Ta sama konfiguracja dawała 0.896 albo 0.928 | `set_seed` przed `build_model` + `cudnn.deterministic`; powtórka daje identyczny wynik |

Oba wykryte przy audycie kodu. Wszystkie wyniki poniżej są po naprawie.

## 2. Porównanie architektur (walidacja)

Fine-tuning = trenowany ostatni blok sieci + klasyfikator; head = tylko klasyfikator.

| Model | Wariant | Val macro-F1 | Wagi MB | CPU ms / zdjęcie | GPU ms / zdjęcie |
| --- | --- | ---: | ---: | ---: | ---: |
| **EfficientNet-B0** | finetune | **0.914** | 16.3 | 12.8 | 4.9 |
| ResNet18 | finetune | 0.910 | 44.8 | 8.3 | 2.9 |
| MobileNetV3-Small | finetune | 0.875 | 6.2 | 5.1 | 3.3 |
| EfficientNet-B0 | head | 0.834 | 16.3 | 14.8 | 5.0 |
| MobileNetV3-Small | head | 0.811 | 6.2 | 3.9 | 3.4 |
| ResNet18 | head | 0.780 | 44.8 | 8.3 | 2.9 |

Wnioski:
- Fine-tuning ostatniego bloku daje +0.06–0.13 macro-F1 względem samej głowy.
- EfficientNet-B0 i ResNet18 są blisko; EfficientNet wybrany, bo ma najlepszy wynik przy ~3× mniejszych wagach. 13 ms na CPU wystarcza na darmowy CPU w HF Spaces.

## 3. Hiperparametry EfficientNet-B0 finetune

Siatka 3 × 2 × 2 = 12 runów (lr × batch × weight decay), seed 42:
- **Batch 16 > 32** — wszystkie 5 najlepszych wyników miało batch 16.
- **Weight decay prawie bez znaczenia** — przy lr 1e-4 wyniki z i bez były identyczne.
- Najlepsze: lr 1e-3 (0.929) i lr 3e-4 (0.925), batch 16, wd 0.

Powtórki top-2 na trzech seedach (42, 1, 2):

| LR | Seed 42 | Seed 1 | Seed 2 | Średnia ± std |
| ---: | ---: | ---: | ---: | ---: |
| **3e-4** | 0.925 | 0.930 | 0.902 | **0.919 ± 0.015** |
| 1e-3 | 0.929 | 0.917 | 0.888 | 0.911 ± 0.022 |

Wybór: **lr 3e-4, batch 16, wd 0** (`configs/final_efficientnet_b0.yaml`) — wyższa średnia i mniejszy rozrzut.
Różnica między ustawieniami (0.008) jest mniejsza niż rozrzut między seedami, więc to wybór stabilniejszego wariantu, nie dowód wyraźnej przewagi.

## 4. Wynik końcowy na zbiorze testowym

Zbiór testowy użyty **jeden raz**, po wszystkich decyzjach.

| Miara | Walidacja | Test |
| --- | ---: | ---: |
| Macro-F1 | 0.925 | **0.893** |
| Accuracy | 0.929 | **0.910** |

Spadek val → test jest oczekiwany: ustawienia były wybierane pod walidację.

![Macierz pomyłek na teście](img/test_confusion_matrix.png)

| Klasa | F1 | Najczęstszy błąd |
| --- | ---: | --- |
| Karton | 0.93 | 5× jako papier |
| Papier | 0.92 | 3× jako trash |
| Plastik | 0.92 | 5× jako szkło |
| Szkło | 0.91 | rzadkie |
| Metal | 0.91 | 3× jako szkło |
| Trash | 0.76 | 3× jako papier |

Wnioski:
- Mylą się pary podobne wizualnie: **karton ↔ papier** (materiał, kolor) i **plastik ↔ szkło** (przezroczystość, odblaski).
- `trash` jest najsłabszy przez małą liczbę przykładów (20 w teście) i brak jednego wyglądu klasy.
- TrashNet to głównie przedmioty na jasnym tle — wynik nie mówi, jak model działa na zdjęciach z innym tłem (osobny test na własnych zdjęciach: do zrobienia).

## Odtworzenie

```bash
python scripts/prepare_split.py
python scripts/train.py --config configs/final_efficientnet_b0.yaml --variant finetune --run-name final/efficientnet_b0
python scripts/evaluate.py
```
