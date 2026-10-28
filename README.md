# explainable-image-classifier (SortWise)

Aplikacja do klasyfikacji pojedynczego odpadu na zdjęciu (6 kategorii z TrashNet) z wizualizacją Grad-CAM — widać, na czym model „patrzy”.

## Po co ten projekt

Cel to end-to-end portfolio wokół CNN: od danych i treningu, przez ocenę jakości i wyjaśnialność, aż po API, prosty interfejs i (w kolejnych etapach) wdrożenie oraz utrzymanie modelu. Nie chodzi tylko o notebook z accuracy, tylko o działający produkt, w którym decyzje ML są mierzalne i da się je powtórzyć.

Robocza nazwa: **SortWise**.

## Wyniki

EfficientNet-B0 (fine-tuning) na zbiorze testowym TrashNet: **macro-F1 0.893, accuracy 0.910**.
Porównanie modeli, dobór hiperparametrów, macierz pomyłek i heatmapy Grad-CAM: [docs/results.md](docs/results.md).
