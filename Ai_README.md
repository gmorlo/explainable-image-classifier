# explainable-image-classifier

## Kontekst dla AI

Ten dokument jest **głównym planem projektu** oraz punktem odniesienia dla AI pomagającego w implementacji. Opisane funkcje są planowane — ich obecność tutaj nie oznacza, że już istnieją w kodzie. Przed zmianami sprawdź rzeczywistą zawartość repozytorium.

Projekt może objąć niemal cały cykl pracy ML Engineera. Największą wartość daje pokazanie, że potrafisz **uzasadnić decyzje, zmierzyć jakość, wdrożyć model i utrzymywać go po wdrożeniu**. Łącząc to z analizą wymagań, architekturą, bazami danych, testowaniem i zarządzaniem projektem, powstaje materiał do rozmów technicznych szerszy niż sam wynik accuracy.

Robocza nazwa aplikacji: **SortWise**. Nazwa repozytorium: `explainable-image-classifier`.

---

## Cel projektu

Zbudować aplikację webową, która klasyfikuje pojedynczy odpad na zdjęciu do jednej z sześciu kategorii i prezentuje wizualizację Grad-CAM. Użytkownik przesyła zdjęcie, widzi wynik, sprawdza obszary wpływające na predykcję i może poprawić błędną kategorię.

**Docelowy rezultat portfolio:**  
„Zbudowałem aplikację klasyfikującą obrazy wraz z odtwarzalnym treningiem, oceną jakości, automatycznym wdrażaniem, monitoringiem i kontrolowaną aktualizacją modelu.”

Priorytetem jest ukończony, mierzalny produkt rozwijany w trzech wydaniach (V1 → V2 → V3), a nie jednoczesna budowa całej infrastruktury.

---

## Zastosowanie biznesowe

Wsparcie użytkownika lub operatora we wstępnym rozpoznawaniu materiału odpadu. System sugeruje kategorię; użytkownik może ją zweryfikować. Potencjalne zastosowanie: etap pomocniczy w procesie sortowania.

MVP rozpoznaje kategorię odpadu. Nie wybiera pojemnika na podstawie lokalnych zasad, nie ocenia zabrudzenia i nie steruje fizyczną sortownią. Skuteczność w rzeczywistym procesie wymaga osobnej walidacji.

---

## Dane

Używamy gotowego zbioru **TrashNet** (Gary Thung, Mindy Yang).

- Źródło: https://github.com/garythung/trashnet
- Oryginalny zbiór: 2527 zdjęć, głównie pojedyncze przedmioty na białym tle.
- Nie zbieramy własnego zbioru treningowego jako warunku startu.
- Datasetu nie dodajemy do Git. Przed treningiem weryfikujemy wersję, etykiety, liczebności, duplikaty i warunki wykorzystania.

| Etykieta źródłowa | Etykieta w interfejsie | Liczba zdjęć |
| --- | --- | --- |
| `cardboard` | Karton | 403 |
| `glass` | Szkło | 501 |
| `metal` | Metal | 410 |
| `paper` | Papier | 594 |
| `plastic` | Plastik | 482 |
| `trash` | Pozostałe odpady | 137 |

Klasa `trash` jest konkretną klasą ze zbioru, nie uniwersalnym koszem na „nieznane obiekty”.

---

## Stack ML — decyzja techniczna

| Element | Decyzja |
| --- | --- |
| Framework | **PyTorch** (aktualna stabilna wersja) + **torchvision** |
| Akceleracja | **CUDA** — trening i lokalna ewaluacja na GPU NVIDIA |
| Inferencja w demo | Preferencyjnie GPU, jeśli dostępne; fallback CPU w kontenerze / Cloud Run |
| Modele do porównania | Co najmniej **dwa** backbony (patrz eksperymenty) |
| Interpretacja | Grad-CAM |

Trening projektujemy pod CUDA od początku: detekcja urządzenia, `pin_memory`, przenoszenie batchy na GPU, zapis metryk czasu treningu/ewaluacji z informacją o sprzęcie. Kod musi działać też na CPU (CI, Cloud Run bez GPU), ale lokalne eksperymenty zakładają kartę NVIDIA.

---

## Eksperymenty: modele i hiperparametry

Porównanie rozwiązań i uzasadnienie wyboru to jeden z pięciu kluczowych elementów portfolio. Nie wystarczy „wziąć ResNet18”.

### Porównanie modeli (macierz architektur)

Porównujemy **co najmniej dwa** modele przy tym samym podziale danych, tym samym preprocessingu walidacyjnym i tej samej procedurze ewaluacji:

| Model | Rola | Poziom fine-tuningu do sprawdzenia |
| --- | --- | --- |
| **ResNet18** | Baseline — prosty, szybki, dobrze znany | (A) zamrożony backbone + trenowana głowa; (B) fine-tuning części/końca sieci |
| **EfficientNet-B0** lub **MobileNetV3-Small** | Alternatywa — inny balans jakość/rozmiar/czas | Te same warianty A/B |

Wynik: tabela z macro-F1, accuracy, czasem treningu, czasem inferencji (GPU i CPU), rozmiarem wag oraz krótkim uzasadnieniem wyboru modelu produkcyjnego.

### Macierz hiperparametrów

Dla wybranego baseline (najpierw ResNet18, wariant A lub B) przeprowadzamy **ograniczoną siatkę** — nie pełny grid wszystkiego naraz. Cel: pokazać metodykę, nie wyczerpać GPU.

Proponowany zakres startowy (do doprecyzowania po pierwszym przebiegu):

| Hiperparametr | Wartości do sprawdzenia |
| --- | --- |
| Learning rate | `1e-3`, `3e-4`, `1e-4` |
| Batch size | `16`, `32` (dostosuj do VRAM) |
| Optimizer | `Adam` vs `SGD + momentum` (jedna oś na raz względem LR) |
| Weight decay | `0`, `1e-4` |
| Epochs / early stopping | stały max + early stopping na walidacyjnym macro-F1 |
| Augmentacje | zestaw minimalny vs rozszerzony (tylko train) |

Zasady:

1. Najpierw porównaj warianty modelu (A/B × 2 architektury) przy **ustalonych** hiperparametrach.
2. Potem grid/random search hiperparametrów **tylko dla finalisty** (lub top-2).
3. Wszystkie runy zapisuj w trackerze eksperymentów (np. MLflow): parametry, metryki, wersja danych, commit, wagi.
4. Wybór ostateczny uzasadnij pomiarami (jakość vs czas vs rozmiar), nie intuicją.

---

## Zakres MVP (produkt V1)

1. Dodanie jednego zdjęcia JPEG/PNG.
2. Walidacja pliku: format, dekodowanie, limity rozmiaru i pikseli.
3. Klasyfikacja jednego dominującego przedmiotu do jednej z sześciu klas.
4. Prezentacja kategorii i wyników dla wszystkich klas.
5. Grad-CAM nałożony na zdjęcie.
6. Zapis analizy z wersją modelu.
7. Potwierdzenie lub poprawienie kategorii.
8. Historia analiz w interfejsie.

W V1 nie implementujemy: multi-object detection, segmentacji, wideo, aplikacji mobilnej, logowania, automatycznego douczania na poprawkach. Feedback użytkownika jest zapisywany, ale nie zmienia wag automatycznie.

---

## Pełny cykl — 16 etapów

Takie podejście odpowiada cyklowi MLOps (dane → trening → walidacja → wdrożenie → monitoring → ponowny trening). Realizujemy stopniowo według wydań V1–V3.

| Etap | Co robisz w projekcie? | Rezultat |
| --- | --- | --- |
| **1. Problem i wymagania** | Użytkownik, scenariusz, 6 klas, ograniczenia, czas odpowiedzi | Specyfikacja, use case’y, kryteria akceptacji |
| **2. Projekt architektury** | Trening, inferencja, API, frontend, storage; uzasadnienie wyborów | Diagram, ERD, krótkie ADR |
| **3. Pozyskanie i wersjonowanie danych** | Pobranie TrashNet, wersja, pochodzenie | Skrypt + manifest z sumami kontrolnymi |
| **4. EDA** | Liczebności, rozdzielczości, uszkodzone pliki, duplikaty, tło | Notebook/raport z wnioskami |
| **5. Przygotowanie danych** | Stały split train/val/test, preprocessing, augmentacje tylko na train | Pipeline bez przecieku |
| **6. Model bazowy i eksperymenty** | Porównanie architektur + macierz hiperparametrów | Tabela eksperymentów i wybór modelu |
| **7. Śledzenie eksperymentów** | Parametry, metryki, dane, commit, wagi | Historia w MLflow (lub równoważnym) |
| **8. Ewaluacja i analiza błędów** | Macro-F1, per-klasa, confusion matrix; papier–karton; własne zdjęcia | Raport jakości z przykładami |
| **9. Wyjaśnialność** | Grad-CAM: przedmiot vs tło | Galeria poprawnych i błędnych predykcji |
| **10. Pakowanie modelu** | Wagi + preprocessing + mapowanie klas = wersjonowany pakiet | Odtwarzalny artefakt do wdrożenia |
| **11. Serwowanie** | FastAPI, walidacja, błędy, load przy starcie | API + docs + pomiary latency |
| **12. Aplikacja i feedback** | Upload, wynik, historia, poprawka kategorii | Demo z oryginalną predykcją i osobną poprawką |
| **13. Testowanie** | Preprocessing, etykiety, API–model–DB, flow użytkownika | Testy automatyczne + raport wydajności |
| **14. CI/CD i wdrożenie** | Lint/testy, obraz Docker, deploy wersji | Powtarzalne wydanie + rollback |
| **15. Monitoring** | Błędy, latency, wersja modelu, rozkład predykcji, zweryfikowane poprawki | Dashboard i reguły alertów |
| **16. Ponowne trenowanie** | Weryfikacja etykiet → nowa wersja danych → kandydat vs produkcja | Kontrolowana aktualizacja modelu |

Źródło koncepcji MLOps: [Google Cloud — MLOps](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning).

---

## Pięć elementów o największej wartości portfolio

1. **Odtwarzalność eksperymentu** — dane, split, kod, konfiguracja, środowisko; druga osoba może powtórzyć run (z uwzględnieniem niedeterministyczności GPU).
2. **Porównanie rozwiązań i uzasadnienie** — np. fine-tuning poprawił macro-F1, ale wydłużył trening; wybór udokumentowany pomiarami. Tracker: [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking).
3. **Bramka jakości przed wdrożeniem** — nowy model dopiero po progach na jakości, klasach, zgodności API i latency; test set tylko do oceny końcowej.
4. **Rollback** — przywrócenie poprzedniego pakietu (wagi + preprocessing + mapowanie klas).
5. **Zamknięcie pętli feedbacku** — poprawka → weryfikacja → nowa wersja danych → kandydat → ocena → dopiero wtedy produkcja.

---

## Monitoring — trzy osobne warstwy

| Rodzaj | Przykładowe pomiary | Wniosek |
| --- | --- | --- |
| **Działanie usługi** | Błędy, p95 latency, pamięć | Czy aplikacja działa sprawnie? |
| **Zmiana danych wejściowych** | Inne tła, jasność, rozkład cech | Czy napływają zdjęcia inne niż trening? |
| **Jakość predykcji** | Macro-F1 na zweryfikowanych etykietach | Czy model naprawdę klasyfikuje gorzej? |

Zmiana danych lub spadek wyników softmax **nie dowodzi** spadku jakości. Do pomiaru jakości potrzeba poprawnych etykiet; same dobrowolne poprawki użytkowników mogą być niereprezentatywne.

---

## Wydania: V1 / V2 / V3

| Wydanie | Zakres | Cel |
| --- | --- | --- |
| **V1 — działający produkt** | Dane, EDA, split, baseline + porównanie 2 modeli / hiperparametry (w rozsądnym zakresie), ewaluacja, Grad-CAM, API, prosty UI, historia i feedback | Działające demo z mierzalną jakością |
| **V2 — warsztat inżynierski** | MLflow, wersjonowanie artefaktów, testy, Docker, CI/CD, wdrożenie, rollback, bramka jakości | Powtarzalne wydania i bezpieczeństwo zmian |
| **V3 — pełna pętla ML** | Monitoring (3 warstwy), zweryfikowany feedback, retrain, porównanie kandydat vs produkcja | Kontrolowana aktualizacja modelu |

**Możliwe rozszerzenia (opisane, nie wymagane w V1–V3):** Kubernetes, feature store, rozproszone trenowanie, rozbudowana orkiestracja, testy A/B. Dla jednego CNN i małego zbioru nie dają jeszcze proporcjonalnej korzyści.

---

## Model i interpretacja wyników

- Dostosuj warstwę wyjściową do 6 klas.
- Zapisuj mapowanie indeksów → etykiety razem z wagami.
- Ten sam deterministyczny preprocessing w ewaluacji i API; augmentacja tylko na train.
- Softmax prezentuj jako wynik modelu — bez kalibracji nie nazywaj go zweryfikowanym prawdopodobieństwem.
- Grad-CAM dla wskazanej klasy i warstwy konwolucyjnej; nie wyłączaj autograd na całej ścieżce heatmapy.
- Model może sklasyfikować nie-odpad do jednej ze znanych klas; próg niskiego wyniku może sugerować ręczną kontrolę, ale nie rozwiązuje open-set.

Grad-CAM wskazuje obszary wpływające na decyzję — nie jest maską obiektu ani dowodem „prawidłowego rozumowania”.

---

## Ocena jakości

- Stratyfikowany split np. **70/15/15**; zapisz ID zdjęć w każdym podzbiorze.
- Duplikaty / powiązane zdjęcia: usuń lub trzymaj w tym samym folcie — bez przecieku.
- Hiperparametry i progi tylko na walidacji; **test wyłącznie do oceny końcowej**.
- Raportuj: accuracy, **macro-F1**, precision/recall per klasa, confusion matrix.
- Przykłady poprawnych i błędnych predykcji + heatmapy.
- Osobny zestaw **własnych zdjęć** (różne tła) — raport oddzielny od testu TrashNet.
- Zmierz czas predykcji i Grad-CAM na opisanym sprzęcie (GPU/CPU).

Nie zakładamy z góry docelowej skuteczności. Wnioski wynikają z pomiarów — szczególnie dla małej klasy `trash` i zdjęć spoza warunków treningowych.

---

## Proponowana architektura

| Element | Odpowiedzialność | Narzędzia |
| --- | --- | --- |
| Moduł ML | Dane, trening, ewaluacja, predykcja, Grad-CAM | PyTorch + CUDA, torchvision, MLflow |
| Backend | Walidacja, analizy, historia, feedback | FastAPI |
| Frontend | Upload, wynik, heatmapa, historia | Do ustalenia |
| Baza | Metadane analiz i poprawki | PostgreSQL |
| Środowisko | Lokalne uruchomienie | Docker / Compose |
| Wdrożenie | Demonstracja | Docelowo GCP Cloud Run |

Przepływ: UI → API → walidacja → model + Grad-CAM → zapis → UI. Poprawka użytkownika jest osobnym rekordem; nie zmienia automatycznie wag.

Na start: jeden backend, inferencja w procesie, model ładowany przy starcie. Kubernetes / mikroserwisy / kolejki — poza MVP.

Przed Cloud Run: trwały storage (np. GCS) + zewnętrzna baza. Lokalny dysk kontenera nie jest magazynem historii. Retencja i usuwanie zdjęć — jawna decyzja przed publikacją.

---

## Proponowany kontrakt API

Propozycja — nie zakładaj, że endpointy już istnieją.

| Metoda i ścieżka | Przeznaczenie |
| --- | --- |
| `GET /health` | Dostępność |
| `POST /analyses` | Analiza zdjęcia i zapis |
| `GET /analyses` | Historia (paginacja) |
| `GET /analyses/{id}` | Pojedyncza analiza |
| `PUT /analyses/{id}/feedback` | Potwierdzona / poprawiona kategoria |

Wynik: id, data, klasa, wyniki wszystkich klas, wersja modelu, czas, sposób pobrania wizualizacji. Feedback oddzielnie od oryginalnej predykcji.

---

## Kolejność implementacji (start → V1)

1. **Dane:** pobranie, QC, manifest, zapis splitu.
2. **EDA:** notebook z wnioskami.
3. **Eksperymenty:** 2 modele × warianty A/B → grid hiperparametrów dla finalisty → wybór z uzasadnieniem.
4. **Predykcja + Grad-CAM** poza webem.
5. **API** z walidacją i load modelu przy starcie.
6. **Interfejs** + historia + feedback.
7. **Dokumentacja** uruchomienia, ograniczeń i rzeczywistych pomiarów.

Każdy etap kończy się weryfikowalnym rezultatem. Nie zaczynaj od rozbudowanej infrastruktury (to V2+).

---

## Kryteria ukończenia V1

- Analiza poprawnego zdjęcia: kategoria, wyniki klas, Grad-CAM.
- Błędny plik → czytelny błąd, bez awarii.
- Analiza w historii; feedback bez nadpisania oryginalnej predykcji.
- Raport z testu TrashNet + osobna ocena własnych zdjęć.
- Tabela porównania modeli / hiperparametrów z uzasadnieniem wyboru.
- Wagi, mapowanie etykiet i preprocessing wersjonowane i odtwarzalne.
- Dokumentacja opisuje rzeczywiste uruchomienie i pomiary — bez wymyślonych wyników.

---

## Zasady pracy dla AI

- Przed implementacją sprawdź pliki, zależności i stan repo. Nie zakładaj istnienia kodu z planu.
- Realizuj najbliższy uzgodniony etap; nie buduj V2/V3 „przy okazji” V1.
- Oddzielaj trening od kodu serwującego API.
- Spójność etykiet i preprocessingu trening ↔ inferencja.
- Nie commituj sekretów, datasetu ani dużych wag. Dokumentuj sposób ich uzyskania.
- Nie używaj przykładowych wyników jako rzeczywistych predykcji.
- Adekwatne testy: mapowanie klas, kontrakt API, walidacja plików, połączenie model–aplikacja.
- Aktualizuj README użytkownika po etapach: co działa, jak uruchomić, ograniczenia.
- Nie dodawaj komend instalacji/uruchamiania, dopóki odpowiadające pliki nie istnieją i nie zostały sprawdzone.
- Trening lokalny zakładaj na **CUDA**; CI i deploy bez GPU muszą mieć ścieżkę CPU.

---

## Decyzje otwarte

- ~~Framework frontendu.~~ **Decyzja (2026-10-28):** demo w **Gradio** (`app.py`), wdrożenie na **HF Spaces** z podziałem dev/prod (branch `dev` → Space dev, `main` → Space prod), wagi w repo modelu na HF Hub. FastAPI + Postgres + Cloud Run przesunięte na później (historia i feedback).
- ~~Dokładny drugi model.~~ Wybrany EfficientNet-B0 — patrz `docs/results.md`.
- ~~Finalny zakres siatki hiperparametrów.~~ Patrz `docs/results.md`.
- Przechowywanie i retencja zdjęć / wizualizacji.
- Próg sugerowania ręcznej weryfikacji (po walidacji).
- Dostęp do historii i koszty publicznego wdrożenia.

**Na teraz:** aplikacja Gradio lokalnie → pakiet modelu na HF Hub → Spaces dev/prod (na razie prywatne).
