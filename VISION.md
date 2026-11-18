# PhotoCraft AI — Vision

## Aktualny stan

Modul Vision zapewnia analize fotografii otwartej w PhotoCraft.

### Komponenty

- Qwen3-VL — wizualna analiza fotografii przez lokalna Ollame.
- Vision Metrics 2.0 — pomiary jasnosci, RGB, nasycenia i stref tonalnych.
- Vision Diagnostics 1.1 — niezalezne ostrzezenia o mozliwych problemach technicznych.
- Vision Recommend — propozycje dostepnych narzedzi PhotoCraft na podstawie katalogu polecen.

### Integracja

PhotoCraft eksportuje aktualny obraz do PNG.
Vision Metrics oblicza parametry obrazu.
Qwen3-VL przygotowuje opis fotografii.
Vision Diagnostics dodaje osobna sekcje diagnostyczna,
niezalezna od odpowiedzi modelu.

Analiza nie wykonuje automatycznych zmian w dokumencie.

### Wykonane testy

- Fotografia grupowa: oryginal i wersja z przebarwieniem.
- Koncert: trzy wersje.
- Plaza: trzy wersje.
- Portret: trzy wersje.

Sprawdzono wykrywanie zmian tonalnych, nasycenia
i wybranych przesuniec kolorystycznych.

### Ograniczenia

- Wyniki diagnostyki sa heurystyczne.
- Ostrzezenie nie oznacza automatycznie bledu fotograficznego.
- Brak ostrzezenia nie oznacza, ze fotografia jest technicznie idealna.
- Qwen3-VL moze nie rozpoznac widocznych problemow kolorystycznych.
- Diagnostyka pojedynczego obrazu nie zna wygladu oryginalu.
- Modul wymaga dalszych testow na zroznicowanych fotografiach.

### Dalszy rozwoj

Priorytetem jest asystent planowania edycji:
uzytkownik opisuje zamierzony efekt, a PhotoCraft AI
proponuje dostepne techniki, warstwy i maski.

W przyszlosci planowana jest mozliwosc wyboru lokalnych
oraz opcjonalnych zewnetrznych modeli AI.
