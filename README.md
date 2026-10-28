# explainable-image-classifier (SortWise)

Aplikacja do klasyfikacji pojedynczego odpadu na zdjęciu (6 kategorii z TrashNet) z wizualizacją Grad-CAM — widać, na czym model „patrzy”.

## Po co ten projekt

Cel to end-to-end portfolio wokół CNN: od danych i treningu, przez ocenę jakości i wyjaśnialność, aż po API, prosty interfejs i (w kolejnych etapach) wdrożenie oraz utrzymanie modelu. Nie chodzi tylko o notebook z accuracy, tylko o działający produkt, w którym decyzje ML są mierzalne i da się je powtórzyć.

Robocza nazwa: **SortWise**.

## Wyniki

EfficientNet-B0 (fine-tuning) na zbiorze testowym TrashNet: **macro-F1 0.893, accuracy 0.910**.
Porównanie modeli, dobór hiperparametrów, macierz pomyłek i heatmapy Grad-CAM: [docs/results.md](docs/results.md).

## Demo lokalnie

Aplikacja bierze model z `artifacts/runs/final/efficientnet_b0` (patrz „Odtworzenie” w [docs/results.md](docs/results.md)); inny katalog runu wskazuje zmienna `SORTWISE_RUN_DIR`.
Jeśli lokalnych wag nie ma, pobiera wersjonowany pakiet modelu z HF Hub (`SORTWISE_MODEL_REPO`, domyślnie `gmorlo/sortwise-efficientnet-b0`, na razie prywatne) w wersji `SORTWISE_MODEL_VERSION` (domyślnie `v1`). Rollback = ustawienie starszego tagu.
Nową wersję modelu publikuje `python scripts/publish_model.py --tag v2` (tagi są niezmienne).

```bash
python app.py   # http://127.0.0.1:7860
```

Wgraj zdjęcie jednego przedmiotu → top-3 klasy (wynik softmax) i heatmapa Grad-CAM dla klasy top-1.
Czas analizy jednego zdjęcia (predykcja + Grad-CAM, mediana z 57 zdjęć): ~44 ms na CPU desktopa, ~35 ms na RTX 3070.
