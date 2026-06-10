# Rozpoznawanie logotypu Apple - projekt POBR

Projekt z rozpoznawania obiektów w obrazach naturalnych. Celem jest wykrywanie logotypu Apple na zdjęciach z niejednorodnym tłem, zgodnie z wymaganiami przedmiotu POBR.

## Założenia

- język: Python,
- biblioteka OpenCV używana tylko do odczytu i zapisu obrazów,
- główna logika przetwarzania napisana w NumPy,
- dane wejściowe znajdują się w `data`,
- wyniki zapisywane są w `results`.

Pipeline obejmuje:

1. wstępne przetwarzanie obrazu,
2. segmentację obiektu z tła,
3. wyznaczanie cech charakterystycznych,
4. identyfikację przez dopasowanie do wzorca kształtu,
5. zapis masek, detekcji i raportów.

## Struktura

```text
.
├── data/                         # zdjęcia wejściowe
├── docs/                         # opis projektu
├── results/                      # wygenerowane wyniki
├── scripts/
│   └── run_detection.py          # uruchomienie całego pipeline'u
├── src/apple_logo_recognition/   # moduły projektu
├── requirements.txt
└── README.md
```

## Instalacja

Utworzenie środowiska:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Uruchomienie

```powershell
$env:PYTHONPATH="src"
python scripts/run_detection.py --data data --output results
```

Opcjonalnie można zmienić próg akceptacji detekcji:

```powershell
python scripts/run_detection.py --data data --output results --min-score 0.50
```

## Wyniki

Po uruchomieniu powstaną:

- `results/annotated/*_detected.jpg` - obrazy z zaznaczonymi detekcjami,
- `results/masks/*_bright.png` i `results/masks/*_dark.png` - maski segmentacji,
- `results/detections.csv` - tabelaryczny raport,
- `results/detections.json` - pełny raport z cechami.

## Moduły

- `io.py` - odczyt i zapis obrazów przez OpenCV,
- `preprocessing.py` - skala szarości, normalizacja, filtracja, progowanie Otsu,
- `morphology.py` - dylatacja, erozja, otwarcie, zamknięcie i wypełnianie dziur,
- `components.py` - własna implementacja etykietowania komponentów spójnych,
- `features.py` - cechy kształtu i dopasowanie do wzorca,
- `detector.py` - segmentacja i klasyfikacja kandydatów,
- `pipeline.py` - przetwarzanie danych i zapis raportów.

Więcej szczegółów znajduje się w [docs/PROJECT_DESCRIPTION.md](docs/PROJECT_DESCRIPTION.md).
