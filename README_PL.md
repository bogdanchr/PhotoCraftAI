# PhotoCraft AI

**Projekt integrujący lokalne narzędzia AI z edytorem graficznym PhotoCraft, rozwijany z myślą o społeczności.**

PhotoCraft AI łączy zewnętrzny edytor PhotoCraft z lokalnymi modelami językowymi i wizyjnymi, procesami ComfyUI oraz interfejsem sterowania opartym na MCP. Celem jest udostępnienie analizy fotografii, tworzenia masek i wybranych operacji edycyjnych w jednym interfejsie Gradio.

> **Stan projektu:** W trakcie rozwoju. Repozytorium zawiera skrypty integracyjne i przykłady konfiguracji, ale **nie zawiera** edytora PhotoCraft, wag modeli AI ani zewnętrznych aplikacji. Projekt powstawał i był testowany w lokalnym środowisku Windows. Nie jest jeszcze instalatorem typu „jedno kliknięcie”.

**[English README](README.md)**

## Możliwości

- **Ogólny czat AI:** interfejs Gradio z Qwen do rozmowy, pytań i zadań związanych z PhotoCraft.
- **Analiza fotografii:** Qwen3-VL może ocenić wyeksportowane zdjęcie i zaproponować poprawki bez automatycznej edycji obrazu.
- **BiRefNet:** wycinanie tła lub obiektów, w tym przetwarzanie zaznaczonego obszaru.
- **SAM3:** segmentacja obiektów na podstawie opisu tekstowego i tworzenie masek przez ComfyUI.
- **Korekty twarzy:** wspomagane tworzenie korekt jasności/kontrastu i masek, które można ręcznie dopracować w edytorze.
- **Integracja MCP:** komunikacja z interfejsem automatyzacji PhotoCraft z użyciem lokalnego katalogu poleceń.
- **Uruchamianie w Windows:** skrypty uruchamiające usługi i opcjonalny skrót klawiaturowy BiRefNet.

Niektóre funkcje wprowadzają zmiany w otwartym dokumencie. Efekty należy sprawdzić w PhotoCraft i odpowiednio zapisać pracę.

## Jak to działa

```text
Interfejs Gradio (app.py)
  |-- Qwen / Qwen3-VL przez Ollama
  |-- Sterowanie PhotoCraft przez MCP
  |-- Mosty BiRefNet i SAM3 --> ComfyUI
  `-- Lokalne katalogi importu/eksportu <--> edytor PhotoCraft
```

PhotoCraft AI to **niezależny projekt integracyjny**. PhotoCraft, ComfyUI, Ollama, modele AI oraz ich licencje pozostają odrębnymi produktami i projektami.

## Wymagania

- Windows (dołączone skrypty korzystają z `.bat` i PowerShell).
- Python i środowisko wirtualne projektu.
- Edytor PhotoCraft ze zgodnym lokalnym interfejsem sterowania.
- ComfyUI z odpowiednimi procesami, węzłami i modelami BiRefNet/SAM3.
- Ollama z odpowiednimi modelami tekstowymi i wizyjnymi (obecna konfiguracja aplikacji wskazuje `qwen3:14b` i `qwen3-vl:8b`; można ją dostosować).
- Oddzielne środowisko Python z `mcp-proxy`, zgodne z konfiguracją skryptu uruchamiającego.

Do lokalnego przetwarzania AI zalecana jest odpowiednia karta graficzna. Dokładne wersje modeli, węzłów i wymagania sprzętowe wymagają jeszcze weryfikacji i opisania.

## Struktura projektu

```text
PhotoCraftAI/
├── app.py                         # Aplikacja Gradio
├── PhotoCraft_AI.bat              # Uruchamianie usług w Windows
├── local_config.example.bat       # Przykładowa konfiguracja
├── requirements.txt              # Zależności Pythona
├── polecenia_photocraft.txt       # Katalog poleceń PhotoCraft
├── comfy_bridge/                 # Mosty i procesy API ComfyUI / PhotoCraft
├── shortcuts/                    # Skrypt skrótu BiRefNet
└── PhotoCraft_BiRefNet_CtrlAltB/  # Opcjonalny instalator skrótu i instrukcja
```

## Wstępna instalacja (Windows)

To **wstępne wskazówki**, a nie zweryfikowana instrukcja instalacji na czystym komputerze.

1. Zainstaluj osobno PhotoCraft, ComfyUI, Ollama oraz potrzebne modele i węzły, korzystając z dokumentacji tych projektów.
2. Sklonuj lub pobierz repozytorium i otwórz terminal w jego głównym katalogu.
3. Utwórz środowisko wirtualne Pythona i zainstaluj zależności:

   ```powershell
   py -m venv venv
   .\venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Skopiuj `local_config.example.bat` jako `local_config.bat` i dostosuj ścieżki do swojego komputera. **Nie publikuj `local_config.bat` ani nie zapisuj w nim haseł i tokenów.**
5. Sprawdź wymagania skryptu uruchamiającego. Token sterowania PhotoCraft powinien znajdować się poza repozytorium, w lokalizacji zgodnej z konfiguracją.
6. Uruchom `PhotoCraft_AI.bat` dopiero po przygotowaniu zewnętrznych programów i wymaganych środowisk Pythona.

Obecny skrypt uruchamiający zakłada określony układ lokalnych instalacji i portów usług. Dalsze prace obejmą poprawę przenośności i pełną listę zależności.

## Konfiguracja i dane lokalne

- `local_config.example.bat` jest publicznym wzorem, a `local_config.bat` zawiera ustawienia danego komputera i jest pomijany przez Git.
- `PHOTOCRAFT_IMPORT_DIR` i `PHOTOCRAFT_EXPORT_DIR` mogą wskazywać katalogi wymiany plików.
- `PHOTOCRAFT_COMFY_PYTHON` może wskazywać interpreter Pythona dostarczony z ComfyUI.
- Nie dodawaj do repozytorium modeli, wyeksportowanych zdjęć, plików tymczasowych, logów, danych uwierzytelniających ani tokenów.
- Katalog poleceń PhotoCraft jest odczytywany lokalnie przez Gradio, ale nie zastępuje zgodnego interfejsu sterowania w samym edytorze.

## Obecne ograniczenia

- Projekt powstawał przede wszystkim na Windows; dołączone skrypty nie obsługują innych systemów.
- Zewnętrzne aplikacje, procesy, węzły i modele nie są dołączone.
- Część operacji wymaga sprawdzenia wizualnego; odpowiedź modelu AI nie gwarantuje prawidłowego wyniku edycji.
- Instrukcja instalacji od zera, pełna lista wersji zależności, tabela zgodności i testy automatyczne są jeszcze do przygotowania.

## Współtworzenie

Po opublikowaniu repozytorium mile widziane będą zgłoszenia błędów, poprawki dokumentacji i propozycje zmian. W zgłoszeniach i zmianach nie umieszczaj prywatnych fotografii, osobistych ścieżek do plików, haseł, tokenów ani cudzych plików modeli.

## Licencja i autorstwo

**Wybór licencji jest jeszcze przed nami.** Dodamy ją po sprawdzeniu zależności i wymagań dotyczących oznaczenia autorstwa innych projektów. PhotoCraft AI nie rości sobie praw do edytora PhotoCraft, ComfyUI, Ollama ani zewnętrznych modeli AI. Obowiązują je odrębne licencje i warunki.
