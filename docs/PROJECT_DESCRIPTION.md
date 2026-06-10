# Opis projektu

Projekt realizuje detekcję logotypu Apple w zdjęciach naturalnych z niejednorodnym tłem.
Pipeline jest zgodny z klasycznym przebiegiem przetwarzania obrazu:

1. Wczytanie obrazu z użyciem OpenCV.
2. Konwersja do skali szarości zaimplementowana ręcznie w NumPy.
3. Normalizacja kontrastu i filtracja uśredniająca.
4. Segmentacja jasnych i ciemnych obszarów progowaniem Otsu.
5. Operacje morfologiczne, wypełnianie dziur i etykietowanie komponentów.
6. Wyznaczanie cech kształtu: pole, proporcje, wypełnienie, obwód względny i momenty Hu.
7. Identyfikacja metodą dopasowania do binarnego wzorca sylwetki jabłka.
8. Zapis masek, obrazów z ramkami oraz raportów CSV/JSON.

## Ograniczenie OpenCV

Zgodnie z wymaganiami OpenCV jest używane wyłącznie do:

- odczytu obrazu: `cv2.imread`,
- zapisu obrazu: `cv2.imwrite`.

Pozostała logika przetwarzania jest zaimplementowana w modułach projektu przy użyciu NumPy i standardowej biblioteki Pythona.

## Dane wejściowe

Folder `data` zawiera trzy zdjęcia naturalne:

- `apple_image_oneInstance.jpeg`,
- `apple_image_multipleInstances.jpeg`,
- `apple_image_peopleBackground.jpeg`.

W danych występują zarówno pojedyncze logotypy, jak i zdjęcie z wieloma instancjami logotypu.
