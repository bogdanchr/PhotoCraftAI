import traceback
import base64
import asyncio
import ast
import json
import re
import urllib.request
import subprocess
import os
import sys
from pathlib import Path
from uuid import uuid4

import gradio as gr
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MCP_URL = "http://127.0.0.1:8766/mcp"
MODEL = "qwen3:14b"
VISION_MODEL = "qwen3-vl:8b-instruct"
BASE_DIR = Path(__file__).resolve().parent

EXPORT_DIR = Path(os.environ.get("PHOTOCRAFT_EXPORT_DIR", str(BASE_DIR / "exports")))
WORK_DIR = Path(os.environ.get("PHOTOCRAFT_IMPORT_DIR", str(BASE_DIR / "import")))
COMMAND_CATALOG_PATH = BASE_DIR / "polecenia_photocraft.txt"

READ_TOOLS = {"session_list", "doc_inspect"}

# Integracja dzialajacego mostu BiRefNet (osobny proces, bez zagniezdzania MCP).
BIREFNET_SCRIPT = BASE_DIR / "comfy_bridge" / "photocraft_comfy_birefnet_v1_4.py"
BIREFNET_SELECTION_SCRIPT = (
    BASE_DIR / "comfy_bridge" / "photocraft_birefnet_selection_v1.py"
)
# Most SAM3 (osobny proces; BiRefNet bez zmian).
SAM3_SCRIPT = BASE_DIR / "comfy_bridge" / "photocraft_sam3_description_v1.py"


SYSTEM_PROMPT = """
Jesteś PhotoCraft AI, asystentem edytora PhotoCraft. Odpowiadaj po polsku.
Ważne: odróżniaj ograniczenia zakresu od globalnego zakazu edycji.
"Nie zmieniaj tła", "nie zapisuj", "nie zmieniaj innych warstw" NIE oznaczają
zakazu wykonania wyraźnie zleconej korekty na wskazanej warstwie.
"Nic nie wykonuj", "niczego nie zmieniaj", "bez zmian" oznaczają brak zgody na edycję.
Nie obiecuj wykonania wieloetapowego procesu, jeśli dostępne narzędzia go nie realizują.
Narzędzia mogą wymagać osobnych komunikatów; raportuj tylko faktycznie wykonane kroki.
PhotoCraft nie jest Photoshopem.
Używaj narzędzi do sprawdzania rzeczywistego stanu edytora.
Po otrzymaniu wyniku narzędzia kontynuuj pierwotne polecenie.
Nie traktuj wyniku JSON jako nowej wiadomości użytkownika.
Liczba dokumentów to długość tablicy session.documents.
session.active to indeks aktywnego dokumentu, liczony od zera.
Liczba warstw to długość tablicy document.layers.
activeLayer jest identyfikatorem warstwy, a nie liczbą warstw.
Nie twierdź, że wykonałeś operację, jeśli jej nie wykonałeś.
Wycinanie obiektu z opisu SAM3 jest osobnym, kontrolowanym mostem.
Po załadowaniu zaznaczenia most podejmuje próbę utworzenia edytowalnej maski
na kopii warstwy. Wynik potwierdzany jest przez odczyt warstw.
Gdy PhotoCraft odmówi automatycznego utworzenia maski, krok pozostaje ręczny.
Wycinanie tła i wycinanie zaznaczonego obszaru BiRefNet to dwie osobne,
deterministyczne funkcje uruchamiane wyłącznie na jawne polecenie. Zaznaczenie
w PhotoCraft jest prostokątem obejmującym; nie gwarantuje wyizolowania jednej
osoby z ciasnej grupy.
Możesz tworzyć nowe dokumenty przez doc_new wyłącznie na wyraźne polecenie.
Możesz dodawać warstwy przez photocraft_layer_add na wyraźne polecenie.
Możesz usuwać pojedyncze warstwy przez photocraft_layer_delete na wyraźne polecenie.
Przed usunięciem sprawdź doc_inspect i odczytaj identyfikator oraz nazwę warstwy.
Nie zgaduj identyfikatorów. Jeśli nazwa jest niejednoznaczna, zapytaj użytkownika.
Nie usuwaj warstwy Background, nie zapisuj ani nie zamykaj dokumentów.
Możesz tworzyć proste kształty przez photocraft_shape_add oraz tekst przez photocraft_text_add,
ale tylko na wyraźne polecenie użytkownika. Dla koła użyj kind ellipse oraz
kwadratowego rect [x,y,średnica,średnica]. Współrzędne liczone od lewego górnego rogu.
Przed rozmieszczeniem elementów sprawdź wymiary dokumentu przez doc_inspect.
Kształty i tekst tworzą osobne warstwy.
Możesz dodać warstwę dopasowania Jasność/Kontrast przez photocraft_brightness_contrast
wyłącznie na wyraźne polecenie użytkownika. Jasność i kontrast to wartości liczbowe,
a nie procenty. Nie zgaduj wartości, gdy użytkownik ich nie podał — zapytaj.
Możesz modyfikować kontrast na AKTUALNIE ZAZNACZONEJ warstwie Jasność/Kontrast
przez photocraft_contrast_existing, tylko na wyraźne polecenie.
Rozróżniaj: „ustaw kontrast na 25” = set 25; „zwiększ kontrast o 10” = delta +10;
„zmniejsz kontrast o 10” = delta -10. Nie twórz przy tym nowej warstwy.
Jeśli użytkownik wyraźnie prosi o NOWĄ warstwę, użyj photocraft_brightness_contrast.
Jeśli prośba nie rozstrzyga, czy zmienić istniejącą czy dodać nową, zapytaj.
Możesz dodać NOWĄ warstwę Poziomy (Levels) przez photocraft_levels_new, wyłącznie na wyraźne polecenie z konkretnymi wartościami punktu czerni, gamma i punktu bieli.
Nie stosuj destrukcyjnego image.adjustments.levels. Nie zgaduj wartości.
Możesz zmieniać istniejącą ZAZNACZONĄ warstwę Levels przez photocraft_levels_existing.
Rozróżniaj ustawienie wartości (set) i zmianę względną (delta).
Nie twórz nowej warstwy przy zmianie istniejących Poziomów.
Gdy użytkownik WYRAŹNIE ŻĄDA gradientu (nie gdy go zabrania), wywołaj WYŁĄCZNIE photocraft_gradient_adjustment; NIE wywołuj najpierw photocraft_brightness_contrast.
Możesz wykonać jedną kompletną korektę z gradientem na masce przez
photocraft_gradient_adjustment: NOWA warstwa Jasność/Kontrast, biała maska,
a następnie gradient liniowy biały->czarny na masce. Wymaga wyraźnej prośby
o nową korektę z gradientem, konkretnych wartości jasności i kontrastu
oraz kierunku (góra-dół, dół-góra, lewo-prawo, prawo-lewo lub ukośnie między przeciwległymi rogami).
Dla przyciemnienia prawego górnego rogu użyj kierunku top_right_bottom_left: biel maski u góry po prawej, czerń na dole po lewej.
Nie stosuj tego narzędzia do edycji istniejącej maski ani warstwy.
Nie wykonuj innych modyfikacji dokumentu.
Możesz chronić fragment zdjęcia na ZAZNACZONEJ istniejącej masce korekty
przez photocraft_mask_protect, tylko na wyraźne polecenie użytkownika.
Wymagaj współrzędnych środka, średnicy i krycia; nie zgaduj położenia.
Nie twórz nowej warstwy, nie zapisuj dokumentu.
Gdy użytkownik prosi o obejrzenie, ocenę lub analizę aktualnego zdjęcia,
użyj photocraft_analyze_photo. Wynik opisuje wyeksportowaną fotografię bez interfejsu programu.
Nie nazywaj PhotoCrafta Photoshopem. Nie wymyślaj szczegółów, których nie potwierdza analiza.
Sama analiza nie upoważnia do zmiany dokumentu.
Narzędzie photocraft_command_search wyszukuje opisy komend w lokalnym katalogu.
Wynik wyszukiwania nie daje prawa do wykonania znalezionej komendy.
Maskę przycinającą można utworzyć tylko na wyraźne polecenie i tylko dla zaznaczonej warstwy korekty.
Zakaz zmiany tła lub zapisu dokumentu nie jest globalnym zakazem edycji.
Lokalne automatyczne malowanie twarzy wymaga jawnych współrzędnych, rozmiaru i jasności; nie zgaduj ich.
Na jawne polecenie przygotowania maski do RĘCZNEGO rozświetlania twarzy wolno utworzyć korektę Jasność/Kontrast, dodać Hide All i clipping, ale NIE malować po masce.
Białą maskę Reveal All na istniejącej warstwie obrazu dodaje tylko jawna komenda; jeśli maska już istnieje, nie zastępuj jej.
Zakaz malowania na masce w takiej prośbie nie oznacza zakazu przygotowania warstwy.
"""

LAYER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "photocraft_layer_add",
            "description": "Dodaj jedną pustą warstwę do aktywnego dokumentu na wyraźne polecenie użytkownika.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Nazwa nowej warstwy"}
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "photocraft_layer_delete",
            "description": "Usuń jedną wskazaną warstwę. Najpierw sprawdź doc_inspect; podaj ID i dokładną nazwę warstwy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_id": {
                        "description": "Dokładny identyfikator z doc_inspect",
                        "type": ["integer", "string"],
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Dokładna nazwa z doc_inspect",
                    },
                },
                "required": ["layer_id", "layer_name"],
                "additionalProperties": False,
            },
        },
    },
]


DRAW_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "photocraft_shape_add",
            "description": "Utwórz nową warstwę kształtu: prostokąt lub elipsę (koło to elipsa z równą szerokością i wysokością). Współrzędne w pikselach.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["rect", "ellipse"]},
                    "rect": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "fill": {"type": "string", "description": "Kolor #rrggbb"},
                    "name": {"type": "string"},
                },
                "required": ["kind", "rect", "fill", "name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "photocraft_text_add",
            "description": "Utwórz nową warstwę tekstową. x,y to pozycja bazowa tekstu w pikselach (y oznacza linię bazową).",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "text": {"type": "string"},
                    "size": {"type": "number", "description": "Rozmiar w punktach"},
                    "color": {"type": "string", "description": "Kolor #rrggbb"},
                    "name": {"type": "string"},
                },
                "required": ["x", "y", "text", "size", "color", "name"],
                "additionalProperties": False,
            },
        },
    },
]


LEVELS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "photocraft_levels_new",
            "description": "Dodaj NOWĄ niedestrukcyjną warstwę Poziomy (Levels). Wymaga wyraźnej prośby i wartości: punkt czerni 0-253, gamma 0.01-9.99, punkt bieli 2-255. Wyjście domyślne 0/255. Nie zmienia istniejących warstw.",
            "parameters": {
                "type": "object",
                "properties": {
                    "inBlack": {"type": "integer", "minimum": 0, "maximum": 253},
                    "gamma": {"type": "number", "minimum": 0.01, "maximum": 9.99},
                    "inWhite": {"type": "integer", "minimum": 2, "maximum": 255},
                },
                "required": ["inBlack", "gamma", "inWhite"],
                "additionalProperties": False,
            },
        },
    },
]


async def run_levels_tool(session, arguments, user_message, used_actions):
    if used_actions.intersection(
        {
            "changed",
            "photocraft_brightness_contrast",
            "photocraft_contrast_existing",
            "photocraft_levels_new",
            "photocraft_levels_existing",
        }
    ):
        return "Odmowa: w tej wiadomości wykonano już zmianę warstw lub korekty."
    if not re.search(r"poziom|levels", user_message, re.I) or not re.search(
        r"dodaj|utw[oó]rz|stw[oó]rz|now[aąejy]?\s+warstw", user_message, re.I
    ):
        return "Odmowa: wymagane jest wyraźne polecenie dodania nowej warstwy Poziomy."
    if not isinstance(arguments, dict) or set(arguments) != {
        "inBlack",
        "gamma",
        "inWhite",
    }:
        return "Odmowa: wymagane są trzy parametry Poziomów."
    black, gamma, white = (arguments[k] for k in ("inBlack", "gamma", "inWhite"))
    if (
        type(black) is not int
        or type(white) is not int
        or type(gamma) not in (int, float)
        or not 0 <= black <= 253
        or not 2 <= white <= 255
        or not 0.01 <= gamma <= 9.99
        or black >= white
    ):
        return "Odmowa: niepoprawne wartości Poziomów (punkt czerni musi być mniejszy od punktu bieli)."
    inspection = await session.call_tool("doc_inspect", arguments={})
    before = json.loads(extract_text(inspection))
    old_layers = find_layers(before)
    if not isinstance(old_layers, list) or not before.get("name"):
        return "Odmowa: nie udało się zweryfikować aktywnego dokumentu."
    old_ids = {x.get("id") for x in old_layers if isinstance(x, dict)}
    used_actions.add("photocraft_levels_new")  # blokada także w razie błędu wykonania
    params = {
        "inBlack": black,
        "gamma": float(gamma),
        "inWhite": white,
        "outBlack": 0,
        "outWhite": 255,
    }
    result = await session.call_tool(
        "command_run",
        arguments={"id": "layer.newAdjustmentLayer.levels", "params": params},
    )
    extract_text(result)
    check = await session.call_tool("doc_inspect", arguments={})
    after = json.loads(extract_text(check))
    new_layers = find_layers(after) or []
    added = [
        x for x in new_layers if isinstance(x, dict) and x.get("id") not in old_ids
    ]
    if (
        after.get("name") != before.get("name")
        or len(added) != 1
        or not isinstance((added[0].get("adjustment") or {}).get("Levels"), dict)
    ):
        return "UWAGA: polecenie wykonano, ale nie potwierdzono jednej nowej warstwy Levels. Sprawdź dokument ręcznie."
    level = added[0]
    master = level["adjustment"]["Levels"].get("master", {})
    if not (
        abs(master.get("in_black", -1) - black / 255) < 0.0001
        and abs(master.get("in_white", -1) - white / 255) < 0.0001
        and abs(master.get("gamma", -1) - gamma) < 0.0001
    ):
        return "UWAGA: utworzono warstwę Levels, ale wartości nie zgadzają się z żądanymi. Sprawdź ręcznie."
    return (
        f"Potwierdzono nową warstwę '{level.get('name')}' (ID {level['id']}): "
        f"punkt czerni {black}, gamma {gamma:g}, punkt bieli {white}, wyjście 0–255. "
        "Nie zapisano dokumentu na dysku."
    )


GRADIENT_ADJUSTMENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "photocraft_gradient_adjustment",
            "description": "Utwórz JEDNĄ nową warstwę Jasność/Kontrast z białą maską i liniowym gradientem biały->czarny na masce. Wymaga jawnego polecenia nowej korekty z gradientem, obu wartości i kierunku. Nie dotyka istniejących masek.",
            "parameters": {
                "type": "object",
                "properties": {
                    "brightness": {"type": "integer", "minimum": -150, "maximum": 150},
                    "contrast": {"type": "integer", "minimum": -50, "maximum": 100},
                    "direction": {
                        "type": "string",
                        "enum": [
                            "top_bottom",
                            "bottom_top",
                            "left_right",
                            "right_left",
                            "top_right_bottom_left",
                            "bottom_left_top_right",
                            "top_left_bottom_right",
                            "bottom_right_top_left",
                        ],
                    },
                },
                "required": ["brightness", "contrast", "direction"],
                "additionalProperties": False,
            },
        },
    },
]


async def run_gradient_adjustment_tool(session, arguments, user_message, used_actions):
    if used_actions.intersection(
        {
            "changed",
            "photocraft_brightness_contrast",
            "photocraft_contrast_existing",
            "photocraft_levels_new",
            "photocraft_levels_existing",
            "photocraft_gradient_adjustment",
        }
    ):
        return "Odmowa: wykonano już zmianę dokumentu w tej wiadomości."
    if not isinstance(arguments, dict) or set(arguments) != {
        "brightness",
        "contrast",
        "direction",
    }:
        return "Odmowa: wymagane jasność, kontrast i kierunek."
    brightness, contrast, direction = (
        arguments["brightness"],
        arguments["contrast"],
        arguments["direction"],
    )
    if (
        type(brightness) is not int
        or type(contrast) is not int
        or not -150 <= brightness <= 150
        or not -50 <= contrast <= 100
        or direction
        not in (
            "top_bottom",
            "bottom_top",
            "left_right",
            "right_left",
            "top_right_bottom_left",
            "bottom_left_top_right",
            "top_left_bottom_right",
            "bottom_right_top_left",
        )
    ):
        return "Odmowa: nieprawidłowe parametry korekty gradientowej."
    if not wants_gradient(user_message) or not re.search(
        r"dodaj|utw[oó]rz|stw[oó]rz|now[aąejy]?\s+warstw", user_message, re.I
    ):
        return (
            "Odmowa: wymagane wyraźne polecenie utworzenia nowej korekty z gradientem."
        )
    if not re.search(r"kontrast|jasno[sś][cć]", user_message, re.I):
        return "Odmowa: brak polecenia korekty Jasność/Kontrast."
    if not re.search(r"g[oó]r|d[oó][lł]|lew|praw", user_message, re.I):
        return "Odmowa: użytkownik musi określić kierunek gradientu."
    # Zapobiega pomyleniu kierunku przez model przy prostej frazie użytkownika.
    words = user_message.lower()
    directions = {
        "top_bottom": r"(?:od\s+g[oó]ry\s+do\s+do[lł]u|z\s+g[oó]ry\s+na\s+d[oó][lł]|g[oó]ra.{0,30}d[oó][lł])",
        "bottom_top": r"(?:od\s+do[lł]u\s+do\s+g[oó]ry|z\s+do[lł]u\s+do\s+g[oó]ry)",
        "left_right": r"(?:od\s+lew(?:ej|a|o)\s+do\s+praw|z\s+lew(?:ej|a|o)\s+na\s+praw)",
        "right_left": r"(?:od\s+praw(?:ej|a|o)\s+do\s+lew|z\s+praw(?:ej|a|o)\s+na\s+lew)",
        # Kierunki ukośne wymagają podania obu przeciwległych rogów.
        "top_right_bottom_left": r"(?:od\s+prawego\s+g[oó]rnego\s+rogu\s+do\s+lewego\s+dolnego\s+rogu)",
        "bottom_left_top_right": r"(?:od\s+lewego\s+dolnego\s+rogu\s+do\s+prawego\s+g[oó]rnego\s+rogu)",
        "top_left_bottom_right": r"(?:od\s+lewego\s+g[oó]rnego\s+rogu\s+do\s+prawego\s+dolnego\s+rogu)",
        "bottom_right_top_left": r"(?:od\s+prawego\s+dolnego\s+rogu\s+do\s+lewego\s+g[oó]rnego\s+rogu)",
    }
    # Nie zgadujemy kierunku: użytkownik musi napisać jednoznacznie „od ... do ...”.
    if not re.search(directions[direction], words):
        return "Odmowa: kierunek podany przez model nie odpowiada jednoznacznie poleceniu. Użyj np. 'od góry do dołu'."
    before = json.loads(
        extract_text(await session.call_tool("doc_inspect", arguments={}))
    )
    layers = find_layers(before)
    width, height = before.get("width"), before.get("height")
    if (
        not before.get("name")
        or not isinstance(layers, list)
        or not layers
        or type(width) is not int
        or type(height) is not int
        or width < 2
        or height < 2
    ):
        return "Odmowa: nie udało się potwierdzić aktywnego dokumentu i jego wymiarów."
    old_ids = {x.get("id") for x in layers if isinstance(x, dict)}
    if len(old_ids) != len(layers):
        return "Odmowa: niejednoznaczne identyfikatory warstw."
    used_actions.add("changed")  # blokada ponowienia również przy błędzie częściowym

    async def inspect():
        return json.loads(
            extract_text(await session.call_tool("doc_inspect", arguments={}))
        )

    # Etap 1: nowa warstwa; weryfikacja identyfikatora, bez polegania na numerze 7.
    extract_text(
        await session.call_tool(
            "command_run",
            arguments={
                "id": "layer.newAdjustmentLayer.brightnessContrast",
                "params": {
                    "brightness": brightness,
                    "contrast": contrast,
                    "legacy": False,
                },
            },
        )
    )
    after_create = await inspect()
    current = find_layers(after_create) or []
    added = [x for x in current if isinstance(x, dict) and x.get("id") not in old_ids]
    if (
        after_create.get("name") != before["name"]
        or len(added) != 1
        or len(current) != len(layers) + 1
    ):
        return "UWAGA: po utworzeniu warstwy stan dokumentu jest niejednoznaczny. Przerwano przed dodaniem maski."
    new_layer = added[0]
    bc = (new_layer.get("adjustment") or {}).get("BrightnessContrast") or {}
    if (
        new_layer.get("kind") != "Adjustment"
        or bc.get("brightness") != brightness
        or bc.get("contrast") != contrast
        or new_layer.get("selected") is not True
        or str(after_create.get("activeLayer")) != str(new_layer.get("id"))
    ):
        return "UWAGA: nowa warstwa nie jest aktywna lub ma inne parametry. Przerwano przed maską."
    layer_id = new_layer["id"]
    if new_layer.get("hasMask") is not True:
        extract_text(
            await session.call_tool(
                "command_run",
                arguments={
                    "id": "layer.layerMask.revealAll",
                    "params": {"layer": layer_id},
                },
            )
        )
    after_mask = await inspect()
    masked = next(
        (x for x in (find_layers(after_mask) or []) if x.get("id") == layer_id), None
    )
    if (
        after_mask.get("name") != before["name"]
        or not masked
        or masked.get("hasMask") is not True
        or masked.get("selected") is not True
        or str(after_mask.get("activeLayer")) != str(layer_id)
    ):
        return "UWAGA: nie potwierdzono maski i aktywnej warstwy. Przerwano przed gradientem."
    mid_x, mid_y = width // 2, height // 2
    points = {
        "top_bottom": ([mid_x, 0], [mid_x, height - 1]),
        "bottom_top": ([mid_x, height - 1], [mid_x, 0]),
        "left_right": ([0, mid_y], [width - 1, mid_y]),
        "right_left": ([width - 1, mid_y], [0, mid_y]),
        "top_right_bottom_left": ([width - 1, 0], [0, height - 1]),
        "bottom_left_top_right": ([0, height - 1], [width - 1, 0]),
        "top_left_bottom_right": ([0, 0], [width - 1, height - 1]),
        "bottom_right_top_left": ([width - 1, height - 1], [0, 0]),
    }
    start, end = points[direction]
    extract_text(
        await session.call_tool(
            "command_run",
            arguments={
                "id": "paint.gradient",
                "params": {
                    "from": start,
                    "to": end,
                    "style": "linear",
                    "stops": [[0, "#ffffff"], [1, "#000000"]],
                    "transparency": [[0, 100], [1, 100]],
                    "reverse": False,
                    "dither": True,
                    "opacity": 100,
                    "mode": "normal",
                    "target": "mask",
                },
            },
        )
    )
    after = await inspect()
    after_layers = find_layers(after) or []
    target = next((x for x in after_layers if x.get("id") == layer_id), None)
    other_ids = {x.get("id") for x in after_layers if isinstance(x, dict)}
    if (
        after.get("name") != before["name"]
        or other_ids != old_ids | {layer_id}
        or not target
        or target.get("hasMask") is not True
    ):
        return "UWAGA: polecenie gradientu wysłano, ale końcowy stan warstw wymaga kontroli."
    return (
        f"Utworzono '{target.get('name')}' (ID {layer_id}), jasność {brightness}, kontrast {contrast}. "
        f"Dodano maskę i wysłano gradient liniowy {direction} ({start} -> {end}) na maskę. "
        "Warstwy zachowane, dokumentu nie zapisano. "
        "Zawartości pikselowej maski nie da się potwierdzić przez doc_inspect — sprawdź ją wizualnie."
    )


EXISTING_LEVELS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "photocraft_levels_existing",
            "description": "Zmień JEDEN parametr ZAZNACZONEJ istniejącej warstwy Levels. set ustawia wartość, delta dodaje do obecnej. Nie tworzy nowej warstwy. Parametry wejścia/wyjścia w skali 0-255, gamma 0.01-9.99.",
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter": {
                        "type": "string",
                        "enum": ["inBlack", "gamma", "inWhite", "outBlack", "outWhite"],
                    },
                    "mode": {"type": "string", "enum": ["set", "delta"]},
                    "value": {"type": "number"},
                },
                "required": ["parameter", "mode", "value"],
                "additionalProperties": False,
            },
        },
    },
]


async def run_existing_levels_tool(session, arguments, user_message, used_actions):
    modifying = {
        "changed",
        "photocraft_brightness_contrast",
        "photocraft_contrast_existing",
        "photocraft_levels_new",
        "photocraft_levels_existing",
    }
    if used_actions.intersection(modifying):
        return "Odmowa: w tej wiadomości wykonano już modyfikację."
    if not isinstance(arguments, dict) or set(arguments) != {
        "parameter",
        "mode",
        "value",
    }:
        return "Odmowa: wymagane parameter, mode i value."
    field, mode, value = arguments["parameter"], arguments["mode"], arguments["value"]
    names = {
        "inBlack": "punkt czerni",
        "gamma": "gamma",
        "inWhite": "punkt bieli",
        "outBlack": "czerń wyjściowa",
        "outWhite": "biel wyjściowa",
    }
    if (
        field not in names
        or mode not in ("set", "delta")
        or type(value) not in (int, float)
    ):
        return "Odmowa: niepoprawny parametr, tryb lub wartość."
    import math

    if not math.isfinite(value):
        return "Odmowa: wartość musi być skończona."
    if not re.search(
        r"poziom|levels|gamm|punkt.{0,12}(czerni|bieli)|czer[nń]|biel",
        user_message,
        re.I,
    ):
        return "Odmowa: brak wyraźnego polecenia zmiany Poziomów."
    if re.search(
        r"now[aąejy]?\s+warstw|dodaj\s+warstw|utw[oó]rz\s+warstw", user_message, re.I
    ):
        return "Odmowa: polecenie dotyczy nowej warstwy."
    if mode == "delta" and not re.search(
        r"zwi[eę]ksz|podnie[sś]|dodaj|zmniejsz|obni[zż]|odejmij|\bo\s*[+-]?\d",
        user_message,
        re.I,
    ):
        return "Odmowa: niejasna zmiana względna."
    if mode == "set" and not re.search(
        r"ustaw|zmie[nń]|na\s*[+-]?\d", user_message, re.I
    ):
        return "Odmowa: niejasne ustawienie wartości."
    data = json.loads(
        extract_text(await session.call_tool("doc_inspect", arguments={}))
    )
    layers = find_layers(data)
    if not isinstance(layers, list) or not data.get("name"):
        return "Odmowa: nie udało się sprawdzić dokumentu."
    selected = [x for x in layers if isinstance(x, dict) and x.get("selected") is True]
    if len(selected) != 1 or selected[0].get("id") != data.get("activeLayer"):
        return "Odmowa: zaznacz dokładnie jedną aktywną warstwę Levels."
    layer = selected[0]
    if layer.get("kind") != "Adjustment" or layer.get("locked") is True:
        return "Odmowa: zaznaczona warstwa nie jest odblokowaną korektą."
    levels = (layer.get("adjustment") or {}).get("Levels")
    if not isinstance(levels, dict) or levels.get("space") != "Rgb":
        return "Odmowa: obsługiwane są tylko warstwy Levels RGB."
    master = levels.get("master")
    keys = {
        "inBlack": "in_black",
        "gamma": "gamma",
        "inWhite": "in_white",
        "outBlack": "out_black",
        "outWhite": "out_white",
    }
    if not isinstance(master, dict) or not all(
        type(master.get(k)) in (int, float) for k in keys.values()
    ):
        return "Odmowa: niekompletne ustawienia Poziomów."
    before = {
        k: float(master[v]) * (1 if k == "gamma" else 255) for k, v in keys.items()
    }
    target = value if mode == "set" else before[field] + value
    if field != "gamma":
        nearest = round(target)
        if abs(target - nearest) > 0.00001:
            return "Odmowa: punkty Poziomów muszą być liczbami całkowitymi."
        target = nearest
    updated = dict(before)
    updated[field] = target
    if not (
        0 <= updated["inBlack"] <= 253
        and 2 <= updated["inWhite"] <= 255
        and updated["inBlack"] < updated["inWhite"]
        and 0 <= updated["outBlack"] <= 255
        and 0 <= updated["outWhite"] <= 255
        and updated["outBlack"] <= updated["outWhite"]
        and 0.01 <= updated["gamma"] <= 9.99
    ):
        return "Odmowa: wartości Poziomów poza zakresem lub punkt czerni >= punktowi bieli."
    if abs(target - before[field]) < 0.000001:
        return f"Bez zmian: {names[field]} ma już wartość {target:g}."
    params = {
        "layer": layer["id"],
        **{k: (v if k == "gamma" else int(round(v))) for k, v in updated.items()},
    }
    old_other = {
        x["id"]: x.get("adjustment")
        for x in layers
        if isinstance(x, dict) and x.get("id") != layer["id"]
    }
    used_actions.add("photocraft_levels_existing")
    extract_text(
        await session.call_tool(
            "command_run", arguments={"id": "layer.setAdjustment", "params": params}
        )
    )
    after = json.loads(
        extract_text(await session.call_tool("doc_inspect", arguments={}))
    )
    after_layers = find_layers(after) or []
    found = [
        x for x in after_layers if isinstance(x, dict) and x.get("id") == layer["id"]
    ]
    if (
        after.get("name") != data["name"]
        or len(after_layers) != len(layers)
        or len(found) != 1
    ):
        return "UWAGA: wykonano polecenie, ale stan dokumentu wymaga ręcznej kontroli."
    after_master = ((found[0].get("adjustment") or {}).get("Levels") or {}).get(
        "master"
    ) or {}
    ok = all(
        abs(after_master.get(keys[k], -999) * (1 if k == "gamma" else 255) - v) < 0.001
        for k, v in updated.items()
    )
    other_ok = all(
        next(
            (x.get("adjustment") for x in after_layers if x.get("id") == ident),
            object(),
        )
        == adj
        for ident, adj in old_other.items()
    )
    if not (ok and other_ok):
        return "UWAGA: wykonano polecenie, ale nie potwierdzono wszystkich parametrów. Sprawdź warstwy ręcznie."
    return (
        f"Potwierdzono zmianę '{layer['name']}' (ID {layer['id']}): "
        f"{names[field]} {before[field]:.3f} → {target:g}. Pozostałe parametry bez zmian. "
        "Nie dodano warstwy ani nie zapisano dokumentu."
    )


EXISTING_ADJUSTMENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "photocraft_contrast_existing",
            "description": "Zmień kontrast AKTUALNIE ZAZNACZONEJ istniejącej warstwy Jasność/Kontrast, bez tworzenia nowej. Tryb set ustawia konkretną wartość, delta dodaje wartość do obecnej (ujemna zmniejsza). Nie zmienia jasności.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["set", "delta"]},
                    "value": {
                        "type": "integer",
                        "description": "Wartość docelowa dla set lub zmiana ze znakiem dla delta",
                    },
                },
                "required": ["mode", "value"],
                "additionalProperties": False,
            },
        },
    },
]


async def run_existing_contrast_tool(session, arguments, user_message, used_actions):
    if used_actions.intersection(
        {
            "photocraft_contrast_existing",
            "photocraft_brightness_contrast",
            "photocraft_levels_new",
            "photocraft_levels_existing",
            "changed",
        }
    ):
        return "Odmowa: w tej wiadomości wykonano już zmianę warstw lub kontrastu."
    if not isinstance(arguments, dict) or set(arguments) != {"mode", "value"}:
        return "Odmowa: nieprawidłowe parametry zmiany kontrastu."
    mode, value = arguments["mode"], arguments["value"]
    if mode not in ("set", "delta") or type(value) is not int:
        return "Odmowa: niepoprawny tryb lub wartość kontrastu."
    if not re.search(r"kontrast", user_message, re.I):
        return "Odmowa: brak wyraźnego polecenia zmiany kontrastu."
    if re.search(
        r"now[aąejy]?\s+warstw|dodaj\s+warstw|utw[oó]rz\s+warstw", user_message, re.I
    ):
        return "Odmowa: użytkownik prosi o nową warstwę, a nie edycję istniejącej."
    if mode == "delta" and not re.search(
        r"zwi[eę]ksz|podnie[sś]|dodaj|zmniejsz|obni[zż]|odejmij|\bo\s*[+-]?\d",
        user_message,
        re.I,
    ):
        return "Odmowa: niejasne polecenie zmiany względnej."
    if mode == "set" and not re.search(
        r"ustaw|zmie[nń]|na\s*[+-]?\d", user_message, re.I
    ):
        return "Odmowa: niejasne polecenie ustawienia kontrastu."
    inspection = await session.call_tool("doc_inspect", arguments={})
    data = json.loads(extract_text(inspection))
    layers = find_layers(data)
    if not isinstance(layers, list):
        return "Odmowa: brak listy warstw."
    selected = [x for x in layers if isinstance(x, dict) and x.get("selected") is True]
    active_id = data.get("activeLayer")
    if len(selected) != 1 or str(selected[0].get("id")) != str(active_id):
        return "Odmowa: zaznacz dokładnie jedną aktywną warstwę Jasność/Kontrast."
    layer = selected[0]
    if layer.get("kind") != "Adjustment" or layer.get("locked") is True:
        return "Odmowa: aktywna warstwa nie jest odblokowaną warstwą dopasowania."
    adj = layer.get("adjustment") or {}
    bc = adj.get("BrightnessContrast") if isinstance(adj, dict) else None
    if not isinstance(bc, dict) or not all(
        k in bc for k in ("brightness", "contrast", "legacy")
    ):
        return "Odmowa: aktywna warstwa nie ma kompletnych parametrów Jasność/Kontrast."
    before = bc["contrast"]
    brightness = bc["brightness"]
    if (
        type(before) not in (int, float)
        or type(brightness) not in (int, float)
        or type(bc["legacy"]) is not bool
    ):
        return "Odmowa: nieprawidłowe parametry istniejącej korekty."
    target = value if mode == "set" else before + value
    if not -50 <= target <= 100:
        return f"Odmowa: docelowy kontrast {target:g} poza zakresem -50 do 100."
    if target == before:
        return f"Bez zmian: kontrast na warstwie {layer['name']} już wynosi {before:g}."
    used_actions.add("photocraft_contrast_existing")
    command = await session.call_tool(
        "command_run",
        arguments={
            "id": "layer.setAdjustment",
            "params": {
                "layer": layer["id"],
                "brightness": brightness,
                "contrast": target,
                "legacy": bc["legacy"],
            },
        },
    )
    extract_text(command)
    check = await session.call_tool("doc_inspect", arguments={})
    after_data = json.loads(extract_text(check))
    after_layers = find_layers(after_data) or []
    match = [
        x for x in after_layers if isinstance(x, dict) and x.get("id") == layer["id"]
    ]
    after_bc = (
        (match[0].get("adjustment") or {}).get("BrightnessContrast", {})
        if len(match) == 1
        else {}
    )
    if after_bc.get("contrast") != target or after_bc.get("brightness") != brightness:
        return "UWAGA: wykonano polecenie, ale weryfikacja parametrów nie powiodła się. Sprawdź warstwę ręcznie."
    return (
        f"Potwierdzono zmianę na warstwie '{layer['name']}' (ID {layer['id']}): "
        f"kontrast {before:g} → {target:g}; jasność {brightness:g} bez zmian. "
        "Nie dodano nowej warstwy i nie zapisano dokumentu."
    )


ADJUSTMENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "photocraft_brightness_contrast",
            "description": "Dodaj nową, niedestrukcyjną warstwę dopasowania Jasność/Kontrast do aktywnego dokumentu. Tylko na wyraźne polecenie użytkownika podające wartości.",
            "parameters": {
                "type": "object",
                "properties": {
                    "brightness": {"type": "integer", "minimum": -150, "maximum": 150},
                    "contrast": {"type": "integer", "minimum": -50, "maximum": 100},
                },
                "required": ["brightness", "contrast"],
                "additionalProperties": False,
            },
        },
    },
]


async def run_adjustment_tool(session, arguments, user_message, used_actions):
    if wants_gradient(user_message):
        return "Odmowa: dla korekty z gradientem uzyj photocraft_gradient_adjustment zamiast zwyklej warstwy."
    if not isinstance(arguments, dict) or set(arguments) != {"brightness", "contrast"}:
        return "Odmowa: wymagane są jasność i kontrast."
    if not re.search(
        r"jasno[sś][cć]|kontrast|warstw[ęa] dopasowania", user_message, re.I
    ):
        return "Odmowa: użytkownik nie polecił korekty jasności ani kontrastu."
    if used_actions.intersection(
        {
            "photocraft_brightness_contrast",
            "photocraft_contrast_existing",
            "photocraft_levels_new",
            "photocraft_levels_existing",
            "changed",
        }
    ):
        return "Odmowa: najwyżej jedna zmiana warstw lub kontrastu na wiadomość."
    brightness, contrast = arguments["brightness"], arguments["contrast"]
    if (
        type(brightness) is not int
        or type(contrast) is not int
        or not -150 <= brightness <= 150
        or not -50 <= contrast <= 100
    ):
        return "Odmowa: wartości poza dopuszczalnym zakresem."
    # Zabezpieczenie przed przypadkową korektą innego dokumentu.
    inspection = await session.call_tool("doc_inspect", arguments={})
    extract_text(inspection)
    result = await session.call_tool(
        "command_run",
        arguments={
            "id": "layer.newAdjustmentLayer.brightnessContrast",
            "params": {"brightness": brightness, "contrast": contrast, "legacy": False},
        },
    )
    text = extract_text(result)
    used_actions.add("photocraft_brightness_contrast")
    return text


def wants_gradient(message):
    """Gradient tylko przy wyraźnej prośbie, nie przy zakazie.

    Obsługuje m.in. „nie dodawaj maski ani gradientu” oraz „bez gradientu”.
    W razie niejednoznaczności nie udostępnia narzędzia zapisu gradientu.
    """
    text = message.lower()
    # Usuń zakazy obejmujące wyliczenia: „nie dodawaj maski ani gradientu”.
    text = re.sub(
        r"\bnie\s+(?:dodawaj|tw[oó]rz|nak[lł]adaj|stosuj|r[oó]b|wykonuj)\b"
        r"[^.!?;\n]*?(?=[.!?;\n]|$)",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\bbez\s+(?:\w+\s+){0,3}?gradient\w*", " ", text, flags=re.I)
    # Samo wspomnienie o gradiencie nie jest poleceniem jego dodania.
    return bool(
        re.search(r"gradient\w*", text, re.I)
        and re.search(
            r"\b(?:dodaj|utw[oó]rz|stw[oó]rz|na[lł][oó][zż]|wykonaj|zastosuj|zrob)\w*\b",
            text,
            re.I,
        )
    )


def valid_hex(value):
    return (
        isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) is not None
    )


async def run_draw_tool(session, name, arguments, user_message, used_actions):
    if not isinstance(arguments, dict):
        return "Odmowa: niepoprawne argumenty."
    if name in used_actions:
        return "Odmowa: najwyżej jeden element danego typu na wiadomość."
    if name == "photocraft_shape_add":
        if not re.search(
            r"narysuj|rysuj|dodaj|utw[oó]rz|stw[oó]rz|ko[lł]o|okr[aą]g|elips|prostok[aą]t|kszta[lł]t",
            user_message,
            re.I,
        ):
            return "Odmowa: brak polecenia utworzenia kształtu."
        if set(arguments) != {"kind", "rect", "fill", "name"}:
            return "Odmowa: niepoprawne parametry kształtu."
        rect = arguments["rect"]
        if (
            arguments["kind"] not in ("rect", "ellipse")
            or not isinstance(rect, list)
            or len(rect) != 4
        ):
            return "Odmowa: niepoprawny kształt lub prostokąt."
        if any(
            isinstance(n, bool)
            or not isinstance(n, (int, float))
            or not -10000 <= n <= 10000
            for n in rect
        ):
            return "Odmowa: współrzędne poza dopuszczalnym zakresem."
        if rect[2] <= 0 or rect[3] <= 0 or not valid_hex(arguments["fill"]):
            return "Odmowa: niepoprawne wymiary lub kolor."
        if (
            not isinstance(arguments["name"], str)
            or not 1 <= len(arguments["name"].strip()) <= 120
        ):
            return "Odmowa: niepoprawna nazwa."
        params = {
            "kind": arguments["kind"],
            "rect": rect,
            "fill": arguments["fill"],
            "name": arguments["name"].strip(),
        }
        result = await session.call_tool(
            "command_run", arguments={"id": "shape.create", "params": params}
        )
    elif name == "photocraft_text_add":
        if not re.search(
            r"napis|tekst|podpis|napisz|dodaj|wstaw|utw[oó]rz", user_message, re.I
        ):
            return "Odmowa: brak polecenia dodania tekstu."
        if set(arguments) != {"x", "y", "text", "size", "color", "name"}:
            return "Odmowa: niepoprawne parametry tekstu."
        if any(
            isinstance(arguments[k], bool)
            or not isinstance(arguments[k], (int, float))
            or not -10000 <= arguments[k] <= 10000
            for k in ("x", "y")
        ):
            return "Odmowa: niepoprawne współrzędne tekstu."
        if (
            isinstance(arguments["size"], bool)
            or not isinstance(arguments["size"], (int, float))
            or not 1 <= arguments["size"] <= 300
        ):
            return "Odmowa: niepoprawny rozmiar tekstu."
        if not valid_hex(arguments["color"]):
            return "Odmowa: niepoprawny kolor tekstu."
        if any(
            not isinstance(arguments[k], str)
            or not 1 <= len(arguments[k].strip()) <= 200
            for k in ("text", "name")
        ):
            return "Odmowa: niepoprawny tekst lub nazwa."
        params = {k: arguments[k] for k in ("x", "y", "text", "size", "color", "name")}
        result = await session.call_tool(
            "command_run", arguments={"id": "type.create", "params": params}
        )
    else:
        return "Narzędzie zablokowane."
    text = extract_text(result)
    used_actions.add(name)
    return text


def ollama_chat(messages, tools, think=True):
    payload = {
        "model": MODEL,
        "stream": False,
        "think": think,
        "messages": messages,
        "tools": tools,
        "options": {"num_ctx": 8192},
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        data = json.load(response)
        print(
            "\nOLLAMA DIAGNOSTYKA:",
            "done_reason =",
            data.get("done_reason"),
            "eval_count =",
            data.get("eval_count"),
            "content =",
            repr((data.get("message") or {}).get("content")),
            "thinking_chars =",
            len((data.get("message") or {}).get("thinking") or ""),
            "tool_calls =",
            len((data.get("message") or {}).get("tool_calls") or []),
            flush=True,
        )
        return data["message"]


def extract_text(result):
    text = "\n".join(part.text for part in result.content if part.type == "text")
    if result.isError:
        raise ValueError("Błąd MCP: " + text)
    return text


def find_layers(data):
    """Znajdź tablicę warstw w odpowiedzi doc_inspect, także przy zagnieżdżeniu."""
    if isinstance(data, dict):
        layers = data.get("layers")
        if isinstance(layers, list):
            return layers
        for value in data.values():
            found = find_layers(value)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = find_layers(value)
            if found is not None:
                return found
    return None


async def run_layer_tool(session, name, arguments, user_message, used_actions):
    if not isinstance(arguments, dict):
        return "Odmowa: niepoprawne argumenty."
    if used_actions:
        return "Odmowa: dla bezpieczeństwa wykonuję najwyżej jedną zmianę warstwy na wiadomość."

    if name == "photocraft_layer_add":
        if not re.search(
            r"dodaj|utw[oó]rz|stw[oó]rz|now[aą] warstw", user_message, re.I
        ):
            return "Odmowa: użytkownik nie polecił wyraźnie dodania warstwy."
        if set(arguments) != {"name"} or not isinstance(arguments["name"], str):
            return "Odmowa: wymagana jest nazwa nowej warstwy."
        layer_name = arguments["name"].strip()
        if not 1 <= len(layer_name) <= 120:
            return "Odmowa: nazwa warstwy musi mieć 1–120 znaków."
        result = await session.call_tool(
            "command_run",
            arguments={"id": "layer.new.layer", "params": {"name": layer_name}},
        )
        text = extract_text(result)
        used_actions.add("changed")
        return text

    if name == "photocraft_layer_delete":
        if not re.search(r"usu[nń]|skasuj|wykasuj", user_message, re.I):
            return "Odmowa: użytkownik nie polecił wyraźnie usunięcia warstwy."
        if set(arguments) != {"layer_id", "layer_name"}:
            return "Odmowa: podaj ID i nazwę warstwy."
        layer_id = arguments["layer_id"]
        layer_name = arguments["layer_name"]
        if isinstance(layer_id, bool) or not isinstance(layer_id, (str, int)):
            return "Odmowa: niepoprawny identyfikator warstwy."
        if not isinstance(layer_name, str) or not layer_name.strip():
            return "Odmowa: brak nazwy warstwy."
        if layer_name.casefold().strip() == "background":
            return "Odmowa: nie usuwam warstwy Background."

        # Sprawdź stan edytora bezpośrednio przed usunięciem.
        inspection = await session.call_tool("doc_inspect", arguments={})
        data = json.loads(extract_text(inspection))
        layers = find_layers(data)
        if layers is None:
            return "Odmowa: nie udało się odczytać listy warstw z doc_inspect."
        matches = [
            layer
            for layer in layers
            if isinstance(layer, dict)
            and str(layer.get("id")) == str(layer_id)
            and layer.get("name") == layer_name
        ]
        if len(matches) != 1:
            return (
                "Odmowa: nie znaleziono dokładnie jednej warstwy o podanym ID i nazwie."
            )
        if matches[0].get("locked") or matches[0].get("isBackground"):
            return "Odmowa: nie usuwam warstwy zablokowanej ani tła."

        result = await session.call_tool(
            "command_run",
            arguments={"id": "layer.delete", "params": {"layer": layer_id}},
        )
        text = extract_text(result)
        used_actions.add("changed")
        return text

    return "Narzędzie zablokowane."


VISION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "photocraft_analyze_photo",
            "description": "Przeanalizuj samą fotografię wyeksportowaną z PhotoCraft do PNG przez model Qwen3-VL. Tylko analiza, bez edycji.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


def vision_chat(image_base64, user_message, metrics_report=None):
    # Krótszy prompt i większy budżet odpowiedzi ograniczają ryzyko,
    # że model zużyje limit na rozumowanie i nie zwróci opisu.
    payload = {
        "model": VISION_MODEL,
        "stream": False,
        "think": False,
        "options": {"num_ctx": 8192, "num_predict": 4096},
        "messages": [
            {
                "role": "user",
                "content": (
                    "/no_think\n"
                    "Jesteś asystentem fotografa pracującego w PhotoCraft. "
                    "Oglądasz sam obraz wyeksportowany z programu, bez paneli i ustawień aparatu. "
                    "Odpowiadaj po polsku, konkretnie, naturalnie i bez zbędnych pochwał. "
                    "Oceniaj wyłącznie to, co można stwierdzić na podstawie obrazu. "
                    "Nie zgaduj parametrów ekspozycji, obiektywu, RAW ani ustawień warstw. "
                    "Nie myl jasnego tła z przepaleniem, jeśli szczegóły są widoczne. "
                    "Nie twierdź, że zdjęcie jest nieostre, jeżeli nie można tego ocenić w tej rozdzielczości. "
                    "Nie proponuj przesuwania postaci w zdjęciu, chyba że wyraźnie opiszesz to jako "
                    "zaawansowany montaż, a nie zwykłą korektę. "
                    "Unikaj ogólników takich jak 'harmonizacja kolorów' bez wskazania konkretnego problemu.\n\n"
                    "Odpowiedz w czterech krótkich częściach:\n"
                    "1. OCENA: po jednym rzeczowym zdaniu o kompozycji, świetle, barwach, "
                    "ostrości i roli tła; gdy brak podstaw do oceny, napisz 'nie da się pewnie ocenić'.\n"
                    "2. MOCNE STRONY: maksymalnie dwa faktyczne atuty.\n"
                    "3. POPRAWKI: podaj od zera do trzech rzeczywiscie uzasadnionych korekt. "
                    "Zero poprawek jest prawidlowym wynikiem. "
                    "Nie proponuj zmian tylko po to, by wypelnic liste. "
                    "Kazda korekta musi wynikac z widocznego problemu. "
                    "Dla kazdej podaj: CO poprawic, DLACZEGO i JAKA technika. "
                    "Wskaz, czy potrzebna jest maska lub praca reczna. "
                    "Nie proponuj przesuwania ludzi, zmieniania twarzy ani montazu. "
                    "Nie proponuj dodatkowego rozmywania tla, jesli wyglada dobrze. "
                    "Odr??niaj niedoswietlone cienie od przepalonych swiatel. "
                    "Nie nazywaj subiektywnych preferencji bledami technicznymi. "
                    "Jesli brak istotnych problemow, napisz: Zdjecie nie wymaga korekty.\n"
                    "4. PODSUMOWANIE: wska\u017c najwa\u017cniejsz\u0105 poprawk\u0119 lub napisz, \u017ce zdj\u0119cie nie wymaga korekty.\n"
                    "Niczego nie wykonuj, nie zapisuj plików i nie zmieniaj dokumentu. "
                    "Całość zwięźle: około 180-280 słów, bez tabeli."
                ),
                "images": [image_base64],
            },
        ],
    }

    # Przekazuj do Vision rzeczywiste polecenie uzytkownika.
    import unicodedata

    normalized = unicodedata.normalize("NFKD", user_message.casefold())
    normalized = "".join(
        ch for ch in normalized if not unicodedata.combining(ch)
    )

    focused_colors = (
        any(x in normalized for x in ("opisz", "ocen", "sprawdz", "analiz"))
        and any(x in normalized for x in ("kolor", "barw", "kontrast"))
        and any(x in normalized for x in (
            "wylacznie", "tylko", "nie oceniaj kompozycji"
        ))
    )

    if focused_colors:
        payload["messages"][0]["content"] = (
            "/no_think\n"
            "Jestes asystentem fotografa. Analizujesz widoczny obraz, "
            "a nie historie edycji. Odpowiadaj po polsku.\n"
            "Oceniaj WYLACZNIE kolory, nasycenie, kontrast i tonalnosc. "
            "Nie opisuj kompozycji, ostrosci ani rozmieszczenia postaci. "
            "Nie proponuj poprawek i niczego nie wykonuj.\n"
            "Sprawdz, czy wystepuja widoczne oznaki zbyt mocnej obrobki: "
            "przesadne nasycenie, nienaturalne odcienie skory, "
            "dominanta barwna, zbyt twardy lub zbyt plaski kontrast, "
            "utrata szczegolow w cieniach i swiatlach.\n"
            "Nie zakladaj z gory, ze obraz jest poprawny ani ze jest zepsuty. "
            "Nie wymyslaj wad, ktorych nie widac. "
            "Jesli nie mozna czegos pewnie ocenic, powiedz to.\n"
            "Odpowiedz krotko w czterech punktach:\n"
            "1. BARWY I NASYCENIE\n"
            "2. KONTRAST I TONALNOSC\n"
            "3. SZCZEGOLY W CIENIACH I SWIATLACH\n"
            "4. WNIOSEK TECHNICZNY\n"
            "Dokladne polecenie uzytkownika:\n" + user_message
        )
    else:
        payload["messages"][0]["content"] += (
            "\n\nDokladne polecenie uzytkownika:\n" + user_message
        )


    # Dane liczbowe sa pomocnicze i nie stanowia automatycznej oceny.
    if metrics_report:
        payload["messages"][0]["content"] += (
            "\n\nPOMIARY TECHNICZNE AKTUALNEGO OBRAZU:\n"
            + metrics_report
            + "\nNie oceniaj poprawnosci balansu bieli jedynie "
              "na podstawie srednich RGB. "
              "Statystyki dotycza calego kadru, nie samej skory. "
              "Nie nazywaj clippingu pewnym, jesli dane pokazuja "
              "tylko piksele bliskie wartosciom granicznym. "
              "Polacz pomiary z ocena wizualna."
        )

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    message = result.get("message") or {}
    answer = (message.get("content") or "").strip()
    if not answer and result.get("done_reason") == "length":
        print("VISION: limit tokenów, ponawiam analizę krótszym promptem.")
        retry_payload = {
            "model": VISION_MODEL,
            "stream": False,
            "think": False,
            "options": {"num_ctx": 8192, "num_predict": 8192},
            "messages": [{
                "role": "user",
                "content": (
                    "/no_think\n"
                    "Przeanalizuj załączoną fotografię po polsku. "
                    "Podaj krótko: kompozycję, światło, kolory, ostrość i tło. "
                    "Wymień maksymalnie trzy konkretne poprawki, "
                    "ale tylko jeśli są potrzebne. "
                    "Zaproponuj techniki korekty w edytorze PhotoCraft. "
                    "Nie zgaduj parametrów aparatu ani ustawień warstw. "
                    "Nie wykonuj żadnych zmian. Maksymalnie 180 słów."
                ),
                "images": [image_base64],
            }],
        }
        retry_request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(retry_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(retry_request, timeout=300) as response:
            retry_result = json.load(response)
        answer = ((retry_result.get("message") or {}).get("content") or "").strip()

    if not answer:
        raise ValueError(
            "Model Vision nie zwrócił opisu "
            f"(done_reason={result.get('done_reason')}, "
            f"thinking={'tak' if message.get('thinking') else 'nie'}, "
            f"eval_count={result.get('eval_count', 'brak')}, "
            f"error={result.get('error', 'brak')})."
        )
    return answer


async def analyze_photo(session, user_message, used_actions):
    if not re.search(
        r"obejrzyj|zobacz|oce[nń]|analiz|zdj[eę]ci|fotografi|obraz|poprawki",
        user_message,
        re.I,
    ):
        return "Odmowa: brak prośby o analizę obrazu."
    if "photocraft_analyze_photo" in used_actions:
        return "Analiza została już wykonana dla tej wiadomości."
    used_actions.add("photocraft_analyze_photo")
    # Eksport aktywnego dokumentu (wszystkie widoczne warstwy) bez paneli GUI.
    # Nazwa jest losowa, żeby równoległe analizy nie nadpisywały sobie plików.
    filename = f"vision_{uuid4().hex}.png"
    export_path = EXPORT_DIR / filename
    if not EXPORT_DIR.is_dir():
        raise ValueError(f"Brak folderu eksportu: {EXPORT_DIR}")

    try:
        result = await session.call_tool("doc_export", arguments={"path": filename})
        extract_text(result)  # Zgłasza błąd, jeśli MCP odmówił eksportu.
        if not export_path.is_file():
            raise ValueError(
                f"MCP zgłosił eksport, ale pliku nie ma w {EXPORT_DIR}. "
                "Sprawdź --automation-write-root w PhotoCraft GUI i MCP."
            )
        # PhotoCraft bridge używa ustawień eksportu GUI i może zwrócić inny format.
        # Qwen3-VL otrzymuje tylko potwierdzony plik PNG.
        with export_path.open("rb") as image_file:
            signature = image_file.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(
                "PhotoCraft wyeksportował plik, ale nie jest to PNG. "
                "Ustaw format PNG w ustawieniach eksportu programu PhotoCraft "
                "i spróbuj ponownie."
            )
        image_base64 = base64.b64encode(export_path.read_bytes()).decode("ascii")

        # Odczytowe pomiary tego samego PNG, ktory trafia do Vision.
        metrics_report = None
        metrics_data = None
        try:
            from vision_metrics import analyze_image
            m = await asyncio.to_thread(analyze_image, export_path)
            metrics_data = m
            rgb = m["mean_rgb"]
            sat = m["saturation"]
            neutral = m["bright_neutral_candidates"]

            lines = [
                f"Rozdzielczosc: {m['size'][0]} x {m['size'][1]}",
                f"Srednie RGB: {m['mean_rgb']}",
                f"Srednia jasnosc: {m['mean_luminance']}/255",
                f"Percentyle jasnosci 1/50/99: "
                f"{m['luminance_p01']}/{m['luminance_p50']}/{m['luminance_p99']}",
                f"Blisko czerni: {m['dark_luminance_pct']}%",
                f"Blisko bieli: {m['bright_luminance_pct']}%",
                f"Skrajne wartosci kanalow: {m['channels']}",
                "STREFY TONALNE:",
            ]

            for name, values in m["tonal_bands"].items():
                lines.append(
                    f"{name}: {values['pct']}%; "
                    f"srednie RGB={values['mean_rgb']}"
                )

            lines.extend([
                f"NASYCENIE: mediana={sat['p50']}, "
                f"p90={sat['p90']}, "
                f"mocne={sat['high_pct']}%",
                f"JASNE WZGLEDNIE NEUTRALNE OBSZARY: "
                f"udzial={neutral['pct']}%, "
                f"RGB={neutral['mean_rgb']}, "
                f"roznice={neutral['channel_differences']}",
                "UWAGA: to statystyki jednego obrazu, "
                "nie porownanie z oryginalem. "
                "Nie dowodza same w sobie bledu koloru.",
            ])

            metrics_report = chr(10).join(lines)
            print("VISION METRICS |", metrics_report, flush=True)
        except Exception as exc:
            print("VISION METRICS | Blad:", repr(exc), flush=True)
        analysis = await asyncio.to_thread(vision_chat, image_base64, user_message, metrics_report)
        try:
            from vision_recommend import match_recommendations_to_catalog

            catalog = load_command_catalog()
            suggestions = match_recommendations_to_catalog(analysis, catalog)

            if suggestions and not suggestions.startswith(
                ("Brak", "Nie wykryto")
            ):
                analysis += (
                    "\n\n---\n"
                    "NARZEDZIA PHOTOCRAFT - PROPOZYCJE (bez wykonywania):\n"
                    + suggestions
                )
        except Exception as exc:
            print(f"VISION RECOMMEND | {exc}")


        # Niezalezna diagnostyka techniczna.
        # Nie zmienia odpowiedzi Qwen ani dokumentu PhotoCraft.
        if metrics_data is not None:
            try:
                from vision_diagnostics import diagnose_single

                diagnosis = diagnose_single(metrics_data)
                issues = diagnosis.get("issues", [])

                diagnostic_lines = [
                    "DIAGNOSTYKA TECHNICZNA (niezalezna od Qwen3-VL):"
                ]

                if issues:
                    for issue in issues:
                        diagnostic_lines.append(
                            f"[{issue['severity'].upper()}] "
                            f"{issue['message']}"
                        )
                        if issue.get("evidence"):
                            diagnostic_lines.append(
                                "Dane: " + issue["evidence"]
                            )
                else:
                    diagnostic_lines.append(
                        "Brak silnych sygnalow wedlug aktualnych heurystyk."
                    )

                if diagnosis.get("note"):
                    diagnostic_lines.append(
                        "Uwaga: " + diagnosis["note"]
                    )

                diagnostic_text = "\n".join(diagnostic_lines)
                analysis += "\n\n---\n" + diagnostic_text

                print(
                    "VISION DIAGNOSTICS |",
                    diagnostic_text,
                    flush=True,
                )
            except Exception as exc:
                print(
                    "VISION DIAGNOSTICS | Blad:",
                    repr(exc),
                    flush=True,
                )

        return analysis
    finally:
        # Usuwamy wyłącznie własny plik tymczasowy, nie dotykamy dokumentu.
        export_path.unlink(missing_ok=True)


MASK_PROTECT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "photocraft_mask_protect",
            "description": "Na ZAZNACZONEJ istniejącej warstwie dopasowania z maską namaluj jedną miękką czarną plamkę na masce, aby osłabić korektę w miejscu kwiatu. Nie tworzy warstwy. Współrzędne w pikselach dokumentu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "Środek ochrony X w pikselach dokumentu",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Środek ochrony Y w pikselach dokumentu",
                    },
                    "size": {
                        "type": "integer",
                        "minimum": 20,
                        "maximum": 1500,
                        "description": "Średnica miękkiego pędzla w pikselach",
                    },
                    "opacity": {
                        "type": "number",
                        "minimum": 0.05,
                        "maximum": 1.0,
                        "description": "Krycie czerni od 0.05 do 1.0",
                    },
                },
                "required": ["x", "y", "size", "opacity"],
                "additionalProperties": False,
            },
        },
    }
]


async def run_mask_protect_tool(session, arguments, user_message, used_actions):
    # Operacja zawsze wymaga wyraźnej prośby; bezpiecznik na powtórzenia.
    if used_actions:
        return "Odmowa: wykonano już operację w tej wiadomości."
    if not re.search(
        r"(?:chro[nń]|ochro[nń]|zabezpiecz|nie przyciemniaj|przywr[oó][cć].{0,30}jasno[sś][cć])",
        user_message,
        re.I,
    ):
        return "Odmowa: brak wyraźnej prośby o ochronę fragmentu zdjęcia."
    if not isinstance(arguments, dict) or set(arguments) != {
        "x",
        "y",
        "size",
        "opacity",
    }:
        return "Odmowa: niekompletne parametry ochrony."
    x, y, size, opacity = (arguments[k] for k in ("x", "y", "size", "opacity"))
    if (
        type(x) is not int
        or type(y) is not int
        or type(size) is not int
        or type(opacity) not in (int, float)
        or not 20 <= size <= 1500
        or not 0.05 <= opacity <= 1.0
    ):
        return "Odmowa: niepoprawne współrzędne, rozmiar lub krycie."
    before = json.loads(
        extract_text(await session.call_tool("doc_inspect", arguments={}))
    )
    layers = find_layers(before)
    width, height = before.get("width"), before.get("height")
    if (
        not before.get("name")
        or not isinstance(layers, list)
        or type(width) is not int
        or type(height) is not int
        or not 0 <= x < width
        or not 0 <= y < height
    ):
        return "Odmowa: brak dokumentu lub współrzędne poza obrazem."
    selected = [
        layer
        for layer in layers
        if isinstance(layer, dict) and layer.get("selected") is True
    ]
    if len(selected) != 1:
        return "Odmowa: zaznacz dokładnie jedną warstwę korekty z maską."
    layer = selected[0]
    if (
        str(before.get("activeLayer")) != str(layer.get("id"))
        or layer.get("kind") != "Adjustment"
        or layer.get("hasMask") is not True
        or layer.get("locked") is True
    ):
        return (
            "Odmowa: aktywna warstwa musi być odblokowaną korektą z istniejącą maską."
        )
    if not isinstance((layer.get("adjustment") or {}).get("BrightnessContrast"), dict):
        return "Odmowa: na razie chronimy tylko maskę korekty Jasność/Kontrast."
    # Blokujemy ponowne wykonanie nawet po częściowym błędzie MCP.
    used_actions.add("changed")
    result = await session.call_tool(
        "command_run",
        arguments={
            "id": "paint.stroke",
            "params": {
                "points": [[x, y, 1.0]],
                "size": size,
                "hardness": 0.0,
                "opacity": float(opacity),
                "flow": 1.0,
                "spacing": 0.1,
                "color": "#000000",
                "mode": "normal",
                "erase": False,
                "target": "mask",
            },
        },
    )
    response = extract_text(result)
    after = json.loads(
        extract_text(await session.call_tool("doc_inspect", arguments={}))
    )
    after_layers = find_layers(after) or []
    same_layer = next(
        (item for item in after_layers if item.get("id") == layer.get("id")), None
    )
    if (
        after.get("name") != before.get("name")
        or not same_layer
        or same_layer.get("hasMask") is not True
        or len(after_layers) != len(layers)
    ):
        return (
            "UWAGA: wysłano malowanie maski, ale stan dokumentu wymaga ręcznej kontroli. Wynik: "
            + response[:500]
        )
    return (
        f"Wysłano jedno miękkie czarne pociągnięcie na maskę warstwy '{layer.get('name')}' "
        f"(ID {layer.get('id')}), środek ({x}, {y}), średnica {size}px, krycie {opacity:.0%}. "
        "Nie dodano warstwy ani nie zapisano dokumentu. Sprawdź wizualnie efekt na zdjęciu. "
        f"MCP: {response[:300]}"
    )


# Operacje na grafikach - pojedyncza, jawnie zlecona zmiana na wiadomość.
IMAGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "photocraft_image_place",
            "description": f"Wstaw JEDEN plik PNG/JPG z katalogu {WORK_DIR} jako osadzoną warstwę. Nie twórz pustej warstwy. Podaj samą nazwę pliku, np. tlo.png.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Nazwa pliku, np. tlo.png",
                    },
                },
                "required": ["filename"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "photocraft_layer_translate",
            "description": "Przesuń JEDNĄ warstwę o dx i dy pikseli. Wymagane dokładne ID z doc_inspect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_id": {"type": ["integer", "string"]},
                    "dx": {"type": "integer"},
                    "dy": {"type": "integer"},
                },
                "required": ["layer_id", "dx", "dy"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "photocraft_layer_scale",
            "description": "Przeskaluj JEDNĄ warstwę do procentu jej aktualnego rozmiaru (np. 80). Wymagane ID z doc_inspect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_id": {"type": ["integer", "string"]},
                    "percent": {"type": "number", "minimum": 1, "maximum": 400},
                },
                "required": ["layer_id", "percent"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "photocraft_layer_reorder",
            "description": "Przenieś warstwę nad lub pod inną warstwę; wymagane dwa ID z doc_inspect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_id": {"type": ["integer", "string"]},
                    "target_id": {"type": ["integer", "string"]},
                    "position": {"type": "string", "enum": ["above", "below"]},
                },
                "required": ["layer_id", "target_id", "position"],
                "additionalProperties": False,
            },
        },
    },
]


async def run_image_tool(session, name, arguments, user_message, used_actions):
    if used_actions:
        return "Odmowa: wykonuję najwyżej jedną zmianę dokumentu na wiadomość."
    if not isinstance(arguments, dict):
        return "Odmowa: nieprawidłowe argumenty."
    before = json.loads(
        extract_text(await session.call_tool("doc_inspect", arguments={}))
    )
    layers = find_layers(before)
    if not isinstance(layers, list) or not before.get("name"):
        return "Odmowa: otwórz najpierw dokument w PhotoCraft."
    ids = {str(x.get("id")): x for x in layers if isinstance(x, dict)}
    params = None
    if name == "photocraft_image_place":
        if not re.search(
            r"wstaw|umie[sś][cć]|dodaj|zaimportuj|osad[zź]|placeEmbedded",
            user_message,
            re.I,
        ):
            return "Odmowa: brak wyraźnego polecenia wstawienia grafiki."
        if set(arguments) != {"filename"} or not isinstance(arguments["filename"], str):
            return "Odmowa: podaj nazwę pliku."
        filename = arguments["filename"].strip()
        if (
            not filename
            or Path(filename).name != filename
            or "\\" in filename
            or "/" in filename
            or Path(filename).suffix.lower()
            not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
        ):
            return "Odmowa: podaj samą nazwę obsługiwanego pliku z folderu roboczego."
        source = WORK_DIR / filename
        if not source.is_file():
            return f"Odmowa: nie znaleziono pliku {source}."
        command = "file.placeEmbedded"
        params = {"path": str(source), "fit": True}
    else:
        layer_id = str(arguments.get("layer_id"))
        if layer_id not in ids or ids[layer_id].get("locked") is True:
            return "Odmowa: nie znaleziono odblokowanej warstwy o podanym ID."
        if name == "photocraft_layer_translate":
            if not re.search(
                r"przesu[nń]|przemie[sś][cć]|move|translate", user_message, re.I
            ):
                return "Odmowa: brak polecenia przesunięcia."
            if set(arguments) != {"layer_id", "dx", "dy"} or any(
                type(arguments[k]) is not int or abs(arguments[k]) > 20000
                for k in ("dx", "dy")
            ):
                return "Odmowa: niepoprawne przesunięcie."
            command = "layer.translate"
            params = {
                "layer": ids[layer_id]["id"],
                "dx": arguments["dx"],
                "dy": arguments["dy"],
            }
        elif name == "photocraft_layer_scale":
            if not re.search(
                r"skaluj|przeskaluj|powi[eę]ksz|zmniejsz|rozmiar", user_message, re.I
            ):
                return "Odmowa: brak polecenia skalowania."
            if (
                set(arguments) != {"layer_id", "percent"}
                or type(arguments["percent"]) not in (int, float)
                or not 1 <= arguments["percent"] <= 400
            ):
                return "Odmowa: niepoprawny procent skali."
            bounds = ids[layer_id].get("bounds")
            if (
                not isinstance(bounds, (list, tuple))
                or len(bounds) != 4
                or not all(type(v) in (int, float) for v in bounds)
            ):
                return "Nie znam granic warstwy z doc_inspect; nie skaluję w ciemno."
            x0, y0, x1, y1 = bounds
            factor = arguments["percent"] / 100
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            command = "edit.transform"
            params = {
                "layer": ids[layer_id]["id"],
                "rect": bounds,
                "matrix": [factor, 0, 0, factor, cx * (1 - factor), cy * (1 - factor)],
                "interpolation": "bicubic",
            }
        elif name == "photocraft_layer_reorder":
            if not re.search(
                r"nad|pod|kolejno[sś][cć]|przenie[sś]|u[lł][oó][zż]", user_message, re.I
            ):
                return "Odmowa: brak polecenia zmiany kolejności."
            target_id = str(arguments.get("target_id"))
            if (
                set(arguments) != {"layer_id", "target_id", "position"}
                or target_id not in ids
                or target_id == layer_id
                or arguments["position"] not in ("above", "below")
            ):
                return "Odmowa: niepoprawne identyfikatory lub położenie."
            command = "layer.moveTo"
            params = {
                "layer": ids[layer_id]["id"],
                "target": ids[target_id]["id"],
                "position": arguments["position"],
            }
        else:
            return "Narzędzie zablokowane."
    used_actions.add("changed")  # blokada ponownego zapisu nawet po częściowym błędzie
    response = extract_text(
        await session.call_tool(
            "command_run", arguments={"id": command, "params": params}
        )
    )
    after = json.loads(
        extract_text(await session.call_tool("doc_inspect", arguments={}))
    )
    after_layers = find_layers(after) or []
    if after.get("name") != before.get("name"):
        return (
            "UWAGA: zmienił się aktywny dokument; sprawdź efekt ręcznie. "
            + response[:400]
        )
    if name == "photocraft_image_place":
        old_ids = set(ids)
        added = [x for x in after_layers if str(x.get("id")) not in old_ids]
        if len(added) != 1:
            return (
                "UWAGA: wysłano wstawienie, ale nie potwierdzono jednej nowej warstwy. "
                + response[:400]
            )
        return f"Wstawiono {filename} jako nową warstwę '{added[0].get('name')}' (ID {added[0].get('id')}). Nie zapisano dokumentu."
    return f"Wysłano {command} dla warstwy {layer_id}. Sprawdź efekt wizualnie. MCP: {response[:400]}"


# Katalog komend: tylko wyszukiwanie i opis, bez wykonywania komend.
COMMAND_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "photocraft_command_search",
        "description": "Wyszukaj dostępne polecenia silnika PhotoCraft w lokalnym katalogu. Narzędzie niczego nie zmienia. Używaj przed planowaniem nowych operacji.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Fragment identyfikatora, etykiety lub parametrów, np. blackWhite, clipping, mask",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def load_command_catalog():
    """Odczytuje JSON lub zapis wyniku MCP command_list z konsoli Windows."""
    if not COMMAND_CATALOG_PATH.is_file():
        raise FileNotFoundError(f"Brak katalogu: {COMMAND_CATALOG_PATH}")
    raw = COMMAND_CATALOG_PATH.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # W pliku z CMD wynik MCP jest tekstem reprezentacji Pythona:
        # TextContent(type='text', text='[...]', annotations=...).
        marker = "TextContent(type='text', text="
        pos = raw.find(marker)
        if pos < 0:
            raise ValueError(
                "Nie rozpoznano formatu katalogu. Oczekiwano JSON lub wyniku MCP TextContent."
            )
        pos += len(marker)
        # ast.literal_eval czyta wyłącznie pierwszy literał string (bez eval).
        import tokenize
        import io

        token = next(
            t
            for t in tokenize.generate_tokens(io.StringIO(raw[pos:]).readline)
            if t.type == tokenize.STRING
        )
        data = json.loads(ast.literal_eval(token.string))
    if isinstance(data, dict):
        data = data.get("commands", data.get("result", data))
    if not isinstance(data, list):
        raise ValueError("Katalog nie zawiera listy poleceń.")
    return [x for x in data if isinstance(x, dict) and isinstance(x.get("id"), str)]


def search_command_catalog(query):
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 100:
        return "Podaj krótką frazę wyszukiwania."
    catalog = load_command_catalog()
    terms = query.casefold().split()
    ranked = []
    for command in catalog:
        ident = command.get("id", "").casefold()
        label = str(command.get("label", "")).casefold()
        haystack = " ".join(
            [
                ident,
                label,
                str(command.get("params", "")).casefold(),
                " ".join(map(str, command.get("menu") or [])).casefold(),
            ]
        )
        if all(t in haystack for t in terms):
            score = 10 * sum(t in ident for t in terms) + 4 * sum(
                t in label for t in terms
            )
            ranked.append((score, command))
    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    results = ranked[:12]
    if not results:
        return f"Katalog: {len(catalog)} komend. Brak wyników dla {query!r}."
    lines = [
        f"Katalog: {len(catalog)} komend. Dopasowania: {len(ranked)} (pokazuję maks. 12).",
        "To są opisy, NIE zezwolenie na wykonanie operacji:",
    ]
    for _, cmd in results:
        lines.append(
            f"- {cmd['id']} | {cmd.get('label', '')} | "
            f"enabled={cmd.get('enabled')} | params={str(cmd.get('params', ''))[:450]}"
        )
    return "\n".join(lines)


# v2.6: transakcja dwuetapowa – nowa korekta B&W, potem clipping.
# Nie jest automatycznym wykonawcą dowolnych komend katalogu.


def global_no_edit(message):
    """Odróżnia zakaz WSZYSTKICH zmian od ochrony tła lub innych warstw.

    Przykłady: 'niczego nie zmieniaj' = zakaz; 'nie zmieniaj tła' = ograniczenie.
    """
    text = message.casefold()
    patterns = (
        r"\bniczego\s+nie\s+(?:zmieniaj|modyfikuj|wykonuj|dodawaj|tw[oó]rz)\b",
        r"\bnic\s+nie\s+(?:zmieniaj|modyfikuj|wykonuj|r[oó]b)\b",
        r"\bnie\s+(?:wykonuj|wprowadzaj|r[oó]b)\s+(?:\w+\s+){0,2}?(?:zmian|operacji|modyfikacji)\b",
        r"\bnie\s+(?:zmieniaj|modyfikuj)\s+(?:niczego|czegokolwiek|dokumentu|obrazu|zdj[eę]cia)\b",
        r"\bbez\s+(?:żadnych\s+)?(?:zmian|modyfikacji|edycji)\b",
        r"\b(?:wy[lł][aą]cznie|tylko)\s+(?:odczyt|analiza|wyszukiwanie|sprawdzenie)\b",
    )
    return any(re.search(p, text, re.I) for p in patterns)


def forbid_specific_action(message, action):
    """Zakazy dotyczące konkretnego działania nie są pozwoleniem na jego wykonanie."""
    text = message.casefold()
    if action == "clip":
        return bool(
            re.search(
                r"\bnie\s+(?:tw[oó]rz|dodawaj|stosuj|wykonuj|zak[lł]adaj)\b[^.!?;\n]{0,65}\b(?:mask\w*\s+przycinaj|clipping)",
                text,
            )
        )
    if action == "gradient_map":
        return bool(
            re.search(
                r"\bnie\s+(?:tw[oó]rz|dodawaj|stosuj|wykonuj)\b[^.!?;\n]{0,65}\bgradient\s*map",
                text,
            )
        )
    if action == "bw":
        return bool(
            re.search(
                r"\bnie\s+(?:tw[oó]rz|dodawaj|stosuj|wykonuj)\b[^.!?;\n]{0,65}\b(?:black\s*(?:&|and)?\s*white|czarno.bia)",
                text,
            )
        )
    return False


def wants_new_bw_clipped(message):
    text = message.casefold()
    create = bool(re.search(r"\b(?:dodaj|utw[oó]rz|stw[oó]rz|zrob|zr[oó]b)\w*", text))
    black_white = bool(
        re.search(
            r"black\s*(?:&|and)?\s*white|blackwhite|czarno.bia|czerni\s+i\s+bieli", text
        )
    )
    clipping = bool(
        re.search(
            r"mask[ęaiy]?\s+przycinaj[aą]c|clipping\s*mask|przypn\w*|ogranicz\w*", text
        )
    )
    new_adjustment = bool(
        re.search(
            r"now[aąejy]?\s+warstw|warstw[ęa]\s+(?:korekt|dopasowania)|korekt[ęay]",
            text,
        )
    )
    no_changes = (
        global_no_edit(message)
        or forbid_specific_action(message, "bw")
        or forbid_specific_action(message, "clip")
    )
    return create and black_white and clipping and new_adjustment and not no_changes


async def run_new_bw_clipped(session, user_message):
    if not wants_new_bw_clipped(user_message):
        return "Odmowa: potrzebne jawne polecenie utworzenia nowej korekty Black & White z maską przycinającą."

    async def inspect():
        return json.loads(
            extract_text(await session.call_tool("doc_inspect", arguments={}))
        )

    before = await inspect()
    layers = find_layers(before)
    if not before.get("name") or not isinstance(layers, list) or len(layers) < 2:
        return "Odmowa: nie znaleziono aktywnego dokumentu z warstwami."
    active = [x for x in layers if isinstance(x, dict) and x.get("selected") is True]
    if len(active) != 1 or str(active[0].get("id")) != str(before.get("activeLayer")):
        return (
            "Odmowa: zaznacz dokładnie jedną warstwę źródłową (np. Layer 3 z muzykami)."
        )
    base = active[0]
    if (
        base.get("kind") == "Adjustment"
        or base.get("locked") is True
        or base.get("isBackground") is True
    ):
        return (
            "Odmowa: zaznacz odblokowaną warstwę obrazu, a nie warstwę korekty lub tło."
        )
    old_ids = [str(x.get("id")) for x in layers if isinstance(x, dict)]
    if len(old_ids) != len(layers) or len(old_ids) != len(set(old_ids)):
        return "Odmowa: niejednoznaczne identyfikatory warstw."
    base_index = next(
        (i for i, x in enumerate(layers) if str(x.get("id")) == str(base.get("id"))), -1
    )
    if base_index < 0:
        return "Odmowa: nie odnaleziono warstwy bazowej."

    # DOMYŚLNE neutralne proporcje Black & White wg katalogu PhotoCraft.
    params = {
        "reds": 40,
        "yellows": 60,
        "greens": 40,
        "cyans": 60,
        "blues": 20,
        "magentas": 80,
        "tint": False,
        "tintColor": "#ffffff",
    }
    # Dopiero od tego miejsca modyfikujemy dokument.
    creation = extract_text(
        await session.call_tool(
            "command_run",
            arguments={"id": "layer.newAdjustmentLayer.blackWhite", "params": params},
        )
    )
    after_create = await inspect()
    current = find_layers(after_create)
    if not isinstance(current, list) or after_create.get("name") != before.get("name"):
        return "UWAGA: wysłano utworzenie B&W, ale stan dokumentu jest niejasny. Nie wykonywano maski."
    new = [
        x for x in current if isinstance(x, dict) and str(x.get("id")) not in old_ids
    ]
    if len(new) != 1 or len(current) != len(layers) + 1:
        return "UWAGA: wysłano utworzenie B&W, ale nie potwierdzono dokładnie jednej nowej warstwy. Maski nie wykonano."
    adjustment = new[0]
    idx = next(
        (
            i
            for i, x in enumerate(current)
            if str(x.get("id")) == str(adjustment.get("id"))
        ),
        -1,
    )
    base_now = next(
        (i for i, x in enumerate(current) if str(x.get("id")) == str(base.get("id"))),
        -1,
    )
    unchanged_order = [
        str(x.get("id")) for x in current if str(x.get("id")) in old_ids
    ] == old_ids
    # Kolejność w doc_inspect: warstwy od góry do dołu.
    if (
        not unchanged_order
        or idx < 0
        or base_now != idx + 1
        or adjustment.get("kind") != "Adjustment"
        or not isinstance(adjustment.get("adjustment"), dict)
        or not adjustment.get("adjustment")
        or adjustment.get("selected") is not True
        or str(after_create.get("activeLayer")) != str(adjustment.get("id"))
    ):
        return (
            "UWAGA: nowa warstwa B&W powstała, ale nie potwierdzono jej położenia "
            "bezpośrednio nad źródłem ani zaznaczenia. Nie wykonano maski. "
            f"Nowe ID: {adjustment.get('id')}. Sprawdź warstwy ręcznie."
        )
    clip_response = extract_text(
        await session.call_tool(
            "command_run",
            arguments={
                "id": "layer.createClippingMask",
                "params": {"layer": adjustment["id"]},
            },
        )
    )
    after = await inspect()
    final = find_layers(after)
    if (
        after.get("name") != before["name"]
        or not isinstance(final, list)
        or [str(x.get("id")) for x in final] != [str(x.get("id")) for x in current]
    ):
        return "UWAGA: wysłano clipping, ale lista warstw uległa zmianie. Sprawdź dokument ręcznie."
    return (
        f"Utworzono Black & White '{adjustment.get('name')}' (ID {adjustment['id']}) "
        f"bezpośrednio nad '{base.get('name')}' (ID {base['id']}). "
        "Wysłano Create Clipping Mask dla nowej korekty. Dokumentu nie zapisano. "
        "Sprawdź wizualnie: tylko warstwa bazowa powinna być czarno-biała. "
        "doc_inspect nie potwierdza stanu samego przycięcia. "
        f"MCP: utworzenie={creation[:80]}, clipping={clip_response[:80]}"
    )


# Bezpieczna operacja: maska przycinająca wyłącznie dla zaznaczonej korekty.
CLIPPING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "photocraft_create_clipping_mask",
            "description": "Utwórz clipping mask dla JEDNEJ aktualnie zaznaczonej warstwy korekty. Wymaga wyraźnego polecenia wykonania. Nie wybieraj innej warstwy.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "photocraft_release_clipping_mask",
            "description": "Zwolnij clipping mask tylko na JEDNEJ aktualnie zaznaczonej warstwie korekty. Wymaga jawnego polecenia zwolnienia. Nie wybieraj innej warstwy.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


async def run_clipping_tool(
    session, arguments, user_message, used_actions, release=False
):
    if used_actions:
        return "Odmowa: w tej wiadomości wykonano już operację."
    if arguments != {}:
        return "Odmowa: maska przycinająca nie przyjmuje argumentów."
    if release:
        if not re.search(
            r"\b(?:zwolnij|zwolni[cć]|usu[nń]|od[lł][aą]cz|release)\w*",
            user_message,
            re.I,
        ):
            return "Odmowa: brak wyraźnego polecenia zwolnienia maski."
    elif not re.search(
        r"\b(?:utw[oó]rz|stw[oó]rz|dodaj|zastosuj|wykonaj|zrob|zr[oó]b)\w*",
        user_message,
        re.I,
    ):
        return "Odmowa: brak wyraźnego polecenia utworzenia maski."
    if not re.search(
        r"mask[ęaiy]?\s+przycinaj[aą]c|clipping\s*mask|mask[ęaiy]?\s+przyci[eę]cia",
        user_message,
        re.I,
    ):
        return "Odmowa: polecenie musi jednoznacznie dotyczyć maski przycinającej."
    if global_no_edit(user_message) or forbid_specific_action(user_message, "clip"):
        return "Odmowa: polecenie zabrania tej operacji lub wszystkich zmian."
    before = json.loads(
        extract_text(await session.call_tool("doc_inspect", arguments={}))
    )
    layers = find_layers(before)
    if not before.get("name") or not isinstance(layers, list) or len(layers) < 2:
        return "Odmowa: brak aktywnego dokumentu z co najmniej dwiema warstwami."
    selected = [
        layer
        for layer in layers
        if isinstance(layer, dict) and layer.get("selected") is True
    ]
    if len(selected) != 1:
        return "Odmowa: zaznacz dokładnie jedną warstwę korekty."
    layer = selected[0]
    if (
        str(layer.get("id")) != str(before.get("activeLayer"))
        or layer.get("kind") != "Adjustment"
        or layer.get("locked") is True
    ):
        return "Odmowa: aktywna warstwa musi być odblokowaną warstwą korekty."
    if not isinstance(layer.get("adjustment"), dict) or not layer["adjustment"]:
        return "Odmowa: nie potwierdzono typu korekty."
    # Nie zmieniamy wyboru ani kolejności warstw. Sprawdź w GUI, co jest pod korektą.
    used_actions.add("changed")  # zabezpiecza również przed ponowieniem po błędzie
    command_id = "layer.releaseClippingMask" if release else "layer.createClippingMask"
    response = extract_text(
        await session.call_tool(
            "command_run",
            arguments={"id": command_id, "params": {"layer": layer["id"]}},
        )
    )
    after = json.loads(
        extract_text(await session.call_tool("doc_inspect", arguments={}))
    )
    after_layers = find_layers(after)
    if (
        after.get("name") != before.get("name")
        or not isinstance(after_layers, list)
        or {str(x.get("id")) for x in after_layers if isinstance(x, dict)}
        != {str(x.get("id")) for x in layers if isinstance(x, dict)}
    ):
        return (
            "UWAGA: wysłano polecenie, ale stan warstw wymaga sprawdzenia ręcznego. MCP: "
            + response[:300]
        )
    operation = "Release Clipping Mask" if release else "Create Clipping Mask"
    return (
        f"Wysłano {operation} dla zaznaczonej korekty '{layer.get('name')}' "
        f"(ID {layer['id']}). Nie zapisano dokumentu. "
        + (
            "Sprawdź w PhotoCraft, czy korekta przestała być przycinana. "
            if release
            else "Sprawdź w PhotoCraft, czy korekta działa tylko na warstwę poniżej. "
        )
        + "doc_inspect nie potwierdza wizualnego efektu. MCP: "
        + response[:300]
    )


def select_tool_group(message):
    """Wąska lista funkcji dla jasnych poleceń; przy niejasnych bezpieczne odczyty.

    Nie wykonuje żadnych operacji. Ostateczne zgody i walidacja są w run_*_tool.
    """
    msg = message.lower()
    has = lambda pattern: bool(re.search(pattern, msg, re.I))
    new_layer = has(r"\b(?:dodaj|utw[oó]rz|stw[oó]rz|wstaw)\b") and has(r"warstw")
    delete_layer = has(r"\b(?:usu[nń]|skasuj|wykasuj)\b") and has(r"warstw")
    levels = has(r"poziom|levels|gamm|punkt.{0,12}(?:czerni|bieli)")
    bc = has(r"jasno[sś][cć]|kontrast")
    gradient = wants_gradient(message)
    # Dwuetapowe zlecenie obsługiwane wcześniej w process_message.
    # Operacje na maskach mają pierwszeństwo przed katalogiem i odczytem warstw.
    if has(r"mask[ęaiy]?\s+przycinaj[aą]c|clipping\s*mask|mask[ęaiy]?\s+przyci[eę]cia"):
        if has(r"\b(?:zwolnij|zwolni[cć]|usu[nń]|od[lł][aą]cz|release)\w*"):
            return [CLIPPING_TOOLS[1]]
        if has(r"\b(?:utw[oó]rz|stw[oó]rz|dodaj|zastosuj|wykonaj|zrob|zr[oó]b)\w*"):
            return [CLIPPING_TOOLS[0]]
    # Zapytania o katalog komend mają pierwszeństwo przed innymi odczytami.
    if has(
        r"polecen|komend|funkcj|katalog|command_list|clipping|blackwhite|czarno.bia"
    ):
        return [COMMAND_SEARCH_TOOL]
    # Odczyt stanu nigdy nie jest poleceniem edycji.
    if re.search(
        r"doc_inspect|activeLayer|identyfikator|identyfikatory|lista warstw|stan (?:aktywnego )?dokumentu|zaznaczon(?:ej|a) warstw",
        message,
        re.I,
    ) and not re.search(
        r"\b(?:przesuń|przesun|zmień|zmien|usuń|usun|dodaj|wstaw|przeskaluj)\b",
        message,
        re.I,
    ):
        return []
    # Import obrazu ma pierwszeństwo przed zwykłym dodawaniem pustej warstwy.
    if has(r"wstaw|umie[sś][cć]|zaimportuj|osad[zź]|placeEmbedded") and has(
        r"\.(?:png|jpe?g|tiff?|webp)\b|grafik|obraz|zdj[eę]ci|plik"
    ):
        return [IMAGE_TOOLS[0]]
    if has(r"przesu[nń]|przemie[sś][cć]") and has(r"warstw"):
        return [IMAGE_TOOLS[1]]
    if has(r"skaluj|przeskaluj|powi[eę]ksz|zmniejsz") and has(r"warstw"):
        return [IMAGE_TOOLS[2]]
    if has(r"warstw") and has(r"kolejno[sś][cć]|\bnad\b|\bpod\b"):
        return [IMAGE_TOOLS[3]]
    # Ochrona kwiatu na istniejącej masce ma pierwszeństwo przed korektami.
    if has(
        r"chro[nń]|ochro[nń]|zabezpiecz|nie przyciemniaj|przywr[oó][cć].{0,30}jasno[sś][cć]"
    ):
        return list(MASK_PROTECT_TOOLS)
    # Priorytet złożonej korekty; tylko jeśli to faktyczne żądanie gradientu.
    if gradient and new_layer and bc:
        return list(GRADIENT_ADJUSTMENT_TOOLS)

    # Pytania o wyglad fotografii kieruj do Vision przed korektami.
    if (
        has(r"opisz|oce[n?]|analiz|sprawd[z?]")
        and has(r"zdj[e?]ci|fotografi|obraz|kolor|barw|kontrast|cieni|[s?]wiat[l?]")
    ):
        return list(VISION_TOOLS)

    if levels and new_layer:
        return list(LEVELS_TOOLS)
    if levels:
        return list(EXISTING_LEVELS_TOOLS)
    if bc and new_layer:
        return list(ADJUSTMENT_TOOLS)
    if has(r"kontrast") and not has(r"jasno[sś][cć]"):
        return list(EXISTING_ADJUSTMENT_TOOLS)
    if delete_layer:
        return [LAYER_TOOLS[1]]
    if new_layer:
        return [LAYER_TOOLS[0]]
    if has(r"narysuj|rysuj|ko[lł]o|okr[aą]g|elips|prostok[aą]t|kszta[lł]t"):
        return [DRAW_TOOLS[0]]
    if has(r"napis|tekst|podpis|napisz") and has(r"dodaj|wstaw|utw[oó]rz|napisz"):
        return [DRAW_TOOLS[1]]
    if has(r"obejrzyj|zobacz|oce[nń]|analiz|zdj[eę]ci|fotografi|poprawki"):
        return list(VISION_TOOLS)
    # Przy niejednoznaczności nie udostępniaj wszystkich funkcji zapisu.
    return []


# Wyszukiwanie lokalne, niezależne od sposobu, w jaki Qwen rozumie pytanie.
def is_catalog_lookup(message):
    msg = message.casefold()
    # Nie przechwytuj polecenia stworzenia nowej korekty z maską.
    if wants_new_bw_clipped(message):
        return False
    if re.search(
        r"\b(?:sprawd[zź]|odczytaj|poka[zż])\w*\s+(?:aktywny\s+)?dokument", msg
    ) and not re.search(r"\bkatalog\b|command_list", msg):
        return False
    return (
        bool(
            re.search(
                r"\b(?:katalog|komend|polecen|wyszukaj|znajd[zź]|identyfikator|parametr)\w*",
                msg,
            )
        )
        and bool(
            re.search(
                r"\b(?:znajd[zź]|wyszukaj|poka[zż]|wymie[nń]|podaj|jakie|lista|katalog|dotycz[aą]c)\w*",
                msg,
            )
        )
        and not bool(
            re.search(
                r"\b(?:wykonaj|zastosuj|utw[oó]rz|stw[oó]rz|zmie[nń]|dodaj|usu[nń])\b(?![^.]*\bnie\b)",
                msg,
            )
        )
    )


def catalog_search_queries(message):
    """Zwróć krótkie zapytania, nie całe zdanie (AND search nie lubi polskich przyimków)."""
    msg = message.strip()
    explicit = re.findall(
        r"\b(?:layer|image|paint|path|file|edit|select|view|filter|type)\.[A-Za-z][A-Za-z0-9_.]*",
        msg,
        re.I,
    )
    topics = [
        (r"gradient\s*map|map[ayę]\s+gradient", "gradientMap"),
        (
            r"hue\s*[/&-]\s*saturation|hue\s+and\s+saturation|barw[ayę]\s+i\s+nasyceni|nasyceni",
            "hueSaturation",
        ),
        (r"gradient\s*fill|wype[lł]nieni\w*\s+gradient", "gradient"),
        (r"paint\.gradient|gradient\w*\s+na\s+mas", "paint.gradient"),
        (
            r"black\s*(?:&|and)?\s*white|blackwhite|czarno.bia|czerni\s+i\s+bieli",
            "blackWhite",
        ),
        (r"clipping|przycinaj[aą]c", "clipping"),
        (
            r"brightness\s*[/&-]\s*contrast|jasno[sś][cć]\s*[i/]\s*kontrast",
            "brightnessContrast",
        ),
        (r"\blevels\b|poziom", "levels"),
        (r"\bcurves\b|krzyw", "curves"),
        (r"color\s*balance|balans\s+kolor", "colorBalance"),
        (r"\bmask\b|mask[ęaiy]", "mask"),
    ]
    found = []
    for item in explicit:
        item = item.rstrip(".")
        if item not in found:
            found.append(item)
    for pattern, query in topics:
        if re.search(pattern, msg, re.I) and query not in found:
            found.append(query)
    # Dla nowych nazw funkcji pobieramy frazę po 'dotyczące' albo cytowany identyfikator.
    if not found:
        candidate = re.search(
            r"(?:dotycz[aą]c\w*|funkcj\w*|komend\w*|polece[nń]\w*)\s+(.+?)(?:[.!?]|\s+podaj\b|\s+niczego\b|$)",
            msg,
            re.I | re.S,
        )
        if candidate:
            phrase = candidate.group(1).strip().strip("\"'` ")
            phrase = re.sub(
                r"^(?:funkcji|komendy|polecenia|dotycz[aą]ce)\s+",
                "",
                phrase,
                flags=re.I,
            )
            if 1 <= len(phrase) <= 80:
                found.append(phrase)
    if found:
        return found[:6]
    return (
        ["gradient"]
        if re.search(r"gradient", msg, re.I)
        else ["mask"] if re.search(r"mask", msg, re.I) else ["layer"]
    )


# v2.8: Jednoetapowa intencja użytkownika -> dwa kontrolowane polecenia MCP.
# Paleta jest propozycją do przetestowania na KOPII dokumentu.
SANTANA_GRADIENT_STOPS = [
    [0.0, "#090909"],
    [0.25, "#301010"],
    [0.52, "#751C19"],
    [0.76, "#BD3D2C"],
    [1.0, "#F07858"],
]


def wants_santana_gradient_clipped(message):
    text = message.casefold()
    catalog = bool(
        re.search(
            r"\b(?:znajd[zź]|wyszukaj|katalog|komend|polecen|identyfikator|parametr)\w*",
            text,
        )
    )
    create = bool(re.search(r"\b(?:dodaj|utw[oó]rz|stw[oó]rz|zrob|zr[oó]b)\w*", text))
    gradient_map = bool(re.search(r"gradient\s*map|map[ęay]\s+gradient", text))
    clipping = bool(
        re.search(
            r"mask\w*\s+przycinaj|clipping|przypn\w*|ogranicz\w*|tylko\s+(?:do|na)\s+(?:muzyk|zespo|warstw)",
            text,
        )
    )
    no_changes = (
        global_no_edit(message)
        or forbid_specific_action(message, "gradient_map")
        or forbid_specific_action(message, "clip")
    )
    return create and gradient_map and clipping and not no_changes and not catalog


async def run_santana_gradient_clipped(session, user_message):
    if not wants_santana_gradient_clipped(user_message):
        return "Odmowa: potrzebne jawne polecenie utworzenia nowej Gradient Map z maską przycinającą."

    async def inspect():
        return json.loads(
            extract_text(await session.call_tool("doc_inspect", arguments={}))
        )

    before = await inspect()
    layers = find_layers(before)
    if not before.get("name") or not isinstance(layers, list) or len(layers) < 2:
        return "Odmowa: brak aktywnego dokumentu z warstwami."
    selected = [x for x in layers if isinstance(x, dict) and x.get("selected") is True]
    if len(selected) != 1 or str(selected[0].get("id")) != str(
        before.get("activeLayer")
    ):
        return "Odmowa: wybierz jedną aktywną warstwę obrazu z muzykami."
    base = selected[0]
    if (
        base.get("kind") == "Adjustment"
        or base.get("locked") is True
        or base.get("isBackground") is True
    ):
        return "Odmowa: zaznacz odblokowaną warstwę obrazu, nie korektę ani tło."
    old_ids = [str(x.get("id")) for x in layers if isinstance(x, dict)]
    if len(old_ids) != len(layers) or len(set(old_ids)) != len(old_ids):
        return "Odmowa: identyfikatory warstw są niejednoznaczne."

    # Tylko tworzenie korekty, nigdy destrukcyjne image.adjustments.gradientMap.
    params = {"stops": SANTANA_GRADIENT_STOPS, "reverse": False, "dither": True}
    creation = extract_text(
        await session.call_tool(
            "command_run",
            arguments={
                "id": "layer.newAdjustmentLayer.gradientMap",
                "params": params,
            },
        )
    )
    created = await inspect()
    now = find_layers(created)
    if not isinstance(now, list) or created.get("name") != before["name"]:
        return "UWAGA: wysłano utworzenie Gradient Map, ale nie można potwierdzić dokumentu. Nie wykonano clipping."
    added = [x for x in now if isinstance(x, dict) and str(x.get("id")) not in old_ids]
    if len(now) != len(layers) + 1 or len(added) != 1:
        return "UWAGA: nie potwierdzono jednej nowej warstwy Gradient Map. Nie wykonano clipping."
    adj = added[0]
    idx = next(
        (i for i, x in enumerate(now) if str(x.get("id")) == str(adj.get("id"))), -1
    )
    base_idx = next(
        (i for i, x in enumerate(now) if str(x.get("id")) == str(base.get("id"))), -1
    )
    same_old_order = [
        str(x.get("id")) for x in now if str(x.get("id")) in old_ids
    ] == old_ids
    if (
        not same_old_order
        or idx < 0
        or base_idx != idx + 1
        or adj.get("kind") != "Adjustment"
        or not isinstance(adj.get("adjustment"), dict)
        or not adj.get("adjustment")
        or adj.get("selected") is not True
        or str(created.get("activeLayer")) != str(adj.get("id"))
    ):
        return (
            "UWAGA: nowa Gradient Map powstała, lecz jej położenie lub stan są niejasne. "
            f"ID: {adj.get('id')}. Nie wykonano clipping; sprawdź dokument ręcznie."
        )
    clip = extract_text(
        await session.call_tool(
            "command_run",
            arguments={
                "id": "layer.createClippingMask",
                "params": {"layer": adj["id"]},
            },
        )
    )
    finished = await inspect()
    final_layers = find_layers(finished)
    if (
        finished.get("name") != before["name"]
        or not isinstance(final_layers, list)
        or [str(x.get("id")) for x in final_layers] != [str(x.get("id")) for x in now]
    ):
        return "UWAGA: wykonano clipping, ale stan warstw po operacji jest niejasny. Sprawdź projekt ręcznie."
    return (
        f"Utworzono Gradient Map '{adj.get('name')}' (ID {adj['id']}) nad "
        f"'{base.get('name')}' (ID {base['id']}) i wysłano Create Clipping Mask. "
        "Paleta: czerń → bordo → czerwień → ciepłe czerwone światła. "
        "Nie zapisano dokumentu. Zweryfikuj wizualnie czerwone tło i wygląd twarzy. "
        "doc_inspect nie potwierdza stanu maski przycinającej. "
        f"MCP: utworzenie={creation[:80]}, clipping={clip[:80]}"
    )


# v3.1 — przygotowanie korekty do RĘCZNEGO malowania po masce.
# Ta funkcja nie wywołuje paint.stroke ani nie wskazuje automatycznie twarzy.
def manual_face_mask_request(message):
    if global_no_edit(message) or is_catalog_lookup(message):
        return None
    msg = message.casefold()
    wants = bool(re.search(r"\b(?:przygotuj|utw[oó]rz|dodaj|stw[oó]rz)\b", msg))
    manual = bool(
        re.search(r"r[eę]czn|sam\s+(?:pomaluj[eę]|zamaluj[eę])|do\s+malowania", msg)
    )
    face = bool(re.search(r"twarz|twarzy|twarze|oblicz", msg))
    mask = bool(re.search(r"mask", msg))
    bright = bool(re.search(r"rozja[sś]ni|roz[sś]wietl|do[sś]wietl|jasno[sś][cć]", msg))
    if not (wants and manual and face and mask and bright):
        return None
    if re.search(
        r"\bnie\s+(?:dodawaj|tw[oó]rz|przygotowuj)\b[^.!?;\n]{0,70}\b(?:warstw|korekt|mask)",
        msg,
    ):
        return None
    # Użytkownik musi podać jasność, żeby nie wybierać jej za niego.
    match = re.search(
        r"\b(?:jasno[sś][cć]|brightness)\s*[:=]?\s*\+?([0-9]{1,3})\b", msg
    )
    return int(match.group(1)) if match else 0


async def prepare_manual_face_mask(session, message, brightness):
    if not 1 <= brightness <= 60:
        return "Nie wykonano zmian. Podaj jasność od +1 do +60, np. 'Przygotuj ręczne rozjaśnianie twarzy z maską, jasność=25'."

    async def inspect():
        return json.loads(
            extract_text(await session.call_tool("doc_inspect", arguments={}))
        )

    before = await inspect()
    layers = find_layers(before)
    if not before.get("name") or not isinstance(layers, list) or len(layers) < 2:
        return "Odmowa: brak aktywnego dokumentu z warstwami."
    selected = [x for x in layers if isinstance(x, dict) and x.get("selected") is True]
    if len(selected) != 1 or str(selected[0].get("id")) != str(
        before.get("activeLayer")
    ):
        return "Odmowa: zaznacz dokładnie warstwę obrazu zespołu (Layer 3)."
    base = selected[0]
    if (
        base.get("kind") == "Adjustment"
        or base.get("locked") is True
        or base.get("isBackground") is True
    ):
        return (
            "Odmowa: zaznacz odblokowaną warstwę obrazu zespołu, nie korektę ani tło."
        )
    original_ids = [str(x.get("id")) for x in layers if isinstance(x, dict)]
    if len(original_ids) != len(layers) or len(set(original_ids)) != len(original_ids):
        return "Odmowa: lista warstw ma niejednoznaczne ID."

    # Tworzenie może się powieść nawet w razie błędu kolejnego kroku.
    creation = extract_text(
        await session.call_tool(
            "command_run",
            arguments={
                "id": "layer.newAdjustmentLayer.brightnessContrast",
                "params": {"brightness": brightness, "contrast": 0, "legacy": False},
            },
        )
    )
    created = await inspect()
    current = find_layers(created)
    if not isinstance(current, list) or created.get("name") != before["name"]:
        return "UWAGA: wysłano utworzenie korekty, ale nie potwierdzono dokumentu. Nie dodawano maski."
    new = [
        x
        for x in current
        if isinstance(x, dict) and str(x.get("id")) not in original_ids
    ]
    if len(new) != 1 or len(current) != len(layers) + 1:
        return "UWAGA: nie potwierdzono jednej nowej korekty. Nie dodawano maski."
    adj = new[0]
    idx = next(
        (i for i, x in enumerate(current) if str(x.get("id")) == str(adj.get("id"))), -1
    )
    baseidx = next(
        (i for i, x in enumerate(current) if str(x.get("id")) == str(base.get("id"))),
        -1,
    )
    bc = (adj.get("adjustment") or {}).get("BrightnessContrast")
    if (
        idx < 0
        or baseidx != idx + 1
        or [str(x.get("id")) for x in current if str(x.get("id")) in original_ids]
        != original_ids
        or adj.get("kind") != "Adjustment"
        or not isinstance(bc, dict)
        or bc.get("brightness") != brightness
        or bc.get("contrast") != 0
        or adj.get("selected") is not True
        or str(created.get("activeLayer")) != str(adj.get("id"))
    ):
        return (
            f"UWAGA: korekta ID {adj.get('id')} powstała, ale nie potwierdzono jej położenia "
            "bezpośrednio nad zespołem i ustawień. Przerwano przed maską."
        )
    ident = adj["id"]
    hide = extract_text(
        await session.call_tool(
            "command_run",
            arguments={
                "id": "layer.layerMask.hideAll",
                "params": {"layer": ident},
            },
        )
    )
    masked = await inspect()
    found = next(
        (x for x in (find_layers(masked) or []) if str(x.get("id")) == str(ident)), None
    )
    if (
        masked.get("name") != before["name"]
        or not found
        or found.get("hasMask") is not True
        or found.get("selected") is not True
        or str(masked.get("activeLayer")) != str(ident)
    ):
        return (
            "UWAGA: wysłano Hide All, ale nie potwierdzono maski i aktywnej korekty. "
            "Nie wykonano clipping. Sprawdź ręcznie, czy maska jest czarna."
        )
    clip = extract_text(
        await session.call_tool(
            "command_run",
            arguments={
                "id": "layer.createClippingMask",
                "params": {"layer": ident},
            },
        )
    )
    final = await inspect()
    if final.get("name") != before["name"] or [
        str(x.get("id")) for x in (find_layers(final) or [])
    ] != [str(x.get("id")) for x in current]:
        return "UWAGA: wysłano clipping, ale końcowy stan warstw wymaga kontroli. Nie malowano po masce."
    return (
        f"Przygotowano '{adj.get('name')}' (ID {ident}) nad '{base.get('name')}' (ID {base['id']}): "
        f"jasność +{brightness}, kontrast 0, Hide All oraz Create Clipping Mask. "
        "Nie malowano po masce. Teraz wybierz MINIATURĘ MASKI nowej korekty i maluj "
        "białym, miękkim pędzlem na twarzach; czarnym cofniesz efekt. "
        "Uwaga: starsze warstwy Gradient Map / Black & White nad tą korektą mogą zmieniać rezultat. "
        "Nie zapisano dokumentu. Stan czerni maski i przycięcia sprawdź wizualnie. "
        f"MCP: create={creation[:50]}, hideAll={hide[:50]}, clipping={clip[:50]}"
    )


# v3.0 — świadome, miejscowe rozświetlanie: zaznacz warstwę zespołu,
# podaj środek obszaru w pikselach dokumentu. Brak automatycznej detekcji twarzy.
FACE_LIGHT_PATTERN = re.compile(
    r"(?:rozja[sś]nij|roz[sś]wietl|do[sś]wietl|rozja[sś]nienie|roz[sś]wietlenie).{0,100}(?:twarz|twarzy|twarze|oblicz)|(?:twarz|twarzy|twarze).{0,100}(?:rozja[sś]nij|roz[sś]wietl|do[sś]wietl)",
    re.I | re.S,
)


def face_light_request(message):
    if global_no_edit(message) or is_catalog_lookup(message):
        return None
    if not FACE_LIGHT_PATTERN.search(message):
        return None
    if re.search(
        r"\bnie\s+(?:rozja[sś]niaj|roz[sś]wietlaj|do[sś]wietlaj)\b", message, re.I
    ):
        return None

    # Współrzędne muszą być jawne. Przyjmujemy wyłącznie opisane wartości.
    def integer(names):
        m = re.search(r"\b(?:" + names + r")\s*[:=]\s*([+-]?\d+)\b", message, re.I)
        return int(m.group(1)) if m else None

    x = integer(r"x")
    y = integer(r"y")
    size = integer(r"rozmiar|srednica|średnica|size")
    brightness = integer(r"jasno[sś][cć]|brightness")
    return (x, y, size, brightness)


async def run_face_light(session, message, args):
    x, y, size, brightness = args
    if any(v is None for v in args):
        return (
            "Nie wykonano zmian. Podaj współrzędne i wartości, np. "
            "'Rozświetl twarz: x=650 y=350 rozmiar=160 jasność=20'. "
            "Współrzędne są w pikselach całego dokumentu."
        )
    if not (20 <= size <= 800 and 1 <= brightness <= 60):
        return "Odmowa: rozmiar 20–800 px, jasność 1–60. Nie wykonano zmian."

    async def inspect():
        return json.loads(
            extract_text(await session.call_tool("doc_inspect", arguments={}))
        )

    before = await inspect()
    layers = find_layers(before)
    w, h = before.get("width"), before.get("height")
    if (
        not before.get("name")
        or not isinstance(layers, list)
        or len(layers) < 2
        or type(w) is not int
        or type(h) is not int
        or not (0 <= x < w and 0 <= y < h)
    ):
        return "Odmowa: brak dokumentu, wymiarów lub punkt poza obrazem. Niczego nie zmieniono."
    selected = [
        layer
        for layer in layers
        if isinstance(layer, dict) and layer.get("selected") is True
    ]
    if len(selected) != 1 or str(selected[0].get("id")) != str(
        before.get("activeLayer")
    ):
        return "Odmowa: zaznacz jedną warstwę zespołu (warstwę obrazu). Niczego nie zmieniono."
    base = selected[0]
    if (
        base.get("kind") == "Adjustment"
        or base.get("locked") is True
        or base.get("isBackground") is True
    ):
        return (
            "Odmowa: zaznacz odblokowaną warstwę obrazu zespołu. Niczego nie zmieniono."
        )
    before_ids = [str(l.get("id")) for l in layers if isinstance(l, dict)]
    if len(before_ids) != len(layers) or len(set(before_ids)) != len(before_ids):
        return "Odmowa: niejednoznaczne ID warstw. Niczego nie zmieniono."
    # Etap 1: nowa korekta bez zmian istniejących warstw.
    creation = extract_text(
        await session.call_tool(
            "command_run",
            arguments={
                "id": "layer.newAdjustmentLayer.brightnessContrast",
                "params": {"brightness": brightness, "contrast": 0, "legacy": False},
            },
        )
    )
    created = await inspect()
    now = find_layers(created)
    if not isinstance(now, list) or created.get("name") != before["name"]:
        return "UWAGA: wysłano tworzenie korekty, ale nie udało się potwierdzić dokumentu. Przerwano."
    added = [
        l for l in now if isinstance(l, dict) and str(l.get("id")) not in before_ids
    ]
    if len(added) != 1 or len(now) != len(layers) + 1:
        return "UWAGA: nie potwierdzono dokładnie jednej nowej warstwy. Przerwano przed malowaniem."
    target = added[0]
    idx = next(
        (i for i, l in enumerate(now) if str(l.get("id")) == str(target.get("id"))), -1
    )
    baseidx = next(
        (i for i, l in enumerate(now) if str(l.get("id")) == str(base.get("id"))), -1
    )
    adj = (target.get("adjustment") or {}).get("BrightnessContrast")
    if (
        idx < 0
        or baseidx != idx + 1
        or [str(l.get("id")) for l in now if str(l.get("id")) in before_ids]
        != before_ids
        or target.get("kind") != "Adjustment"
        or not isinstance(adj, dict)
        or adj.get("brightness") != brightness
        or adj.get("contrast") != 0
        or target.get("selected") is not True
        or str(created.get("activeLayer")) != str(target.get("id"))
    ):
        return (
            "UWAGA: utworzono korektę, ale jej parametry, położenie lub zaznaczenie "
            "nie są potwierdzone. Przerwano przed maską i malowaniem."
        )
    ident = target["id"]
    # Etap 2: czarna maska. Sprawdź maskę i aktywne zaznaczenie przed malowaniem.
    mask_result = extract_text(
        await session.call_tool(
            "command_run",
            arguments={"id": "layer.layerMask.hideAll", "params": {"layer": ident}},
        )
    )
    masked = await inspect()
    mask_layers = find_layers(masked) or []
    current = next((l for l in mask_layers if str(l.get("id")) == str(ident)), None)
    if (
        masked.get("name") != before["name"]
        or not current
        or current.get("hasMask") is not True
        or str(masked.get("activeLayer")) != str(ident)
        or current.get("selected") is not True
    ):
        return "UWAGA: nie potwierdzono czarnej maski i aktywnej korekty. Przerwano przed malowaniem."
    # Etap 3: przycięcie do zespołu. Nie zgaduj warstwy docelowej.
    clip_result = extract_text(
        await session.call_tool(
            "command_run",
            arguments={"id": "layer.createClippingMask", "params": {"layer": ident}},
        )
    )
    clip_state = await inspect()
    if (
        clip_state.get("name") != before["name"]
        or str(clip_state.get("activeLayer")) != str(ident)
        or [str(l.get("id")) for l in (find_layers(clip_state) or [])]
        != [str(l.get("id")) for l in now]
    ):
        return "UWAGA: wysłano clipping, ale nie potwierdzono układu warstw. Przerwano przed malowaniem."
    # Etap 4: odsłonięcie jasności białym miękkim pędzlem tylko w zadanym obszarze.
    brush_result = extract_text(
        await session.call_tool(
            "command_run",
            arguments={
                "id": "paint.stroke",
                "params": {
                    "points": [[x, y, 1.0]],
                    "size": size,
                    "hardness": 0.0,
                    "opacity": 0.65,
                    "flow": 1.0,
                    "spacing": 0.1,
                    "color": "#ffffff",
                    "mode": "normal",
                    "erase": False,
                    "target": "mask",
                },
            },
        )
    )
    after = await inspect()
    if after.get("name") != before["name"] or [
        str(l.get("id")) for l in (find_layers(after) or [])
    ] != [str(l.get("id")) for l in now]:
        return "UWAGA: pędzel wysłano, ale końcowy stan warstw wymaga kontroli."
    return (
        f"Utworzono jasną korektę '{target.get('name')}' (ID {ident}), jasność +{brightness}, "
        f"czarną maskę i clipping nad '{base.get('name')}' (ID {base['id']}). "
        f"Wysłano biały miękki pędzel na maskę: x={x}, y={y}, rozmiar={size}px, "
        "krycie 65%. Dokumentu nie zapisano. Sprawdź efekt wzrokowo; "
        "doc_inspect nie potwierdza pikseli maski ani samego clipping. "
        f"MCP: create={creation[:50]}, hideAll={mask_result[:50]}, clip={clip_result[:50]}, stroke={brush_result[:50]}"
    )


# v3.2 — samodzielne dodanie bialej maski do AKTYWNEJ warstwy obrazu.
# Nie maluje, nie wykonuje gradientu, nie zmienia zaznaczenia.
def wants_reveal_all_on_image(message):
    if global_no_edit(message) or is_catalog_lookup(message):
        return False
    text = message.casefold()
    explicit = bool(
        re.search(r"\b(?:dodaj|utw[oó]rz|stw[oó]rz|za[lł][oó][zż])\w*", text)
    )
    reveal = bool(
        re.search(
            r"reveal\s*all|bia[lł][aą]\s+mask[ęa]|mask[ęa]\s+(?:bia[lł][aą]|warstwy)",
            text,
        )
    )
    target = bool(
        re.search(
            r"layer\s*3|warstw\w*\s+zespo[lł]u|warstw\w*\s+(?:obraz\w*|muzyk\w*)|zaznaczon\w*\s+warstw",
            text,
        )
    )
    restricted = bool(
        re.search(
            r"\bnie\s+(?:dodawaj|tw[oó]rz|zak[lł]adaj|wykonuj)\b[^.!?;\n]{0,75}\bmask",
            text,
        )
    )
    return explicit and reveal and target and not restricted


async def run_reveal_all_on_image(session, message):
    async def inspect():
        return json.loads(
            extract_text(await session.call_tool("doc_inspect", arguments={}))
        )

    before = await inspect()
    layers = find_layers(before)
    if not before.get("name") or not isinstance(layers, list):
        return "Odmowa: brak aktywnego dokumentu i listy warstw."
    selected = [x for x in layers if isinstance(x, dict) and x.get("selected") is True]
    if len(selected) != 1 or str(selected[0].get("id")) != str(
        before.get("activeLayer")
    ):
        return "Odmowa: zaznacz jedną warstwę obrazu (Layer 3 z muzykami)."
    layer = selected[0]
    if (
        "layer 3" in message.casefold()
        and layer.get("name", "").casefold() != "layer 3"
    ):
        return f"Odmowa: zaznaczono '{layer.get('name')}', a polecenie dotyczy Layer 3."
    if (
        layer.get("kind") == "Adjustment"
        or layer.get("locked") is True
        or layer.get("isBackground") is True
        or not isinstance(layer.get("id"), (int, str))
    ):
        return "Odmowa: wybierz odblokowaną warstwę obrazu, nie korektę ani tło."
    if layer.get("hasMask") is True:
        return f"Bez zmian: warstwa '{layer.get('name')}' (ID {layer['id']}) ma już maskę. Nie zastępuję jej."
    if layer.get("hasMask") is not False:
        return "Odmowa: doc_inspect nie potwierdza, czy warstwa ma maskę. Nie nadpisuję jej w ciemno."
    ids_before = [str(x.get("id")) for x in layers if isinstance(x, dict)]
    if len(ids_before) != len(layers) or len(set(ids_before)) != len(ids_before):
        return "Odmowa: niejednoznaczna lista warstw."
    response = extract_text(
        await session.call_tool(
            "command_run",
            arguments={
                "id": "layer.layerMask.revealAll",
                "params": {"layer": layer["id"]},
            },
        )
    )
    after = await inspect()
    new_layers = find_layers(after)
    if (
        after.get("name") != before["name"]
        or not isinstance(new_layers, list)
        or [str(x.get("id")) for x in new_layers] != ids_before
    ):
        return "UWAGA: wysłano Reveal All, ale układ warstw wymaga ręcznej kontroli."
    updated = next(
        (x for x in new_layers if str(x.get("id")) == str(layer["id"])), None
    )
    if not updated or updated.get("hasMask") is not True:
        return "UWAGA: wysłano Reveal All, ale doc_inspect nie potwierdził utworzenia maski. Sprawdź PhotoCraft."
    return (
        f"Dodano białą maskę Reveal All do '{layer.get('name')}' (ID {layer['id']}). "
        "Nie użyto pędzla ani gradientu. Nie zapisano dokumentu. "
        "Kliknij miniaturę maski i sprawdź ją w panelu Layers. "
        f"MCP: {response[:150]}"
    )


# v3.4 — gradient istniejącej maski; jawne współrzędne dokumentu ALBO procenty.
# Obsługujemy polecenia deterministycznie, bez przekazywania ich modelowi Qwen.
def parse_lower_edge_fade(message):
    if global_no_edit(message) or is_catalog_lookup(message):
        return None
    msg = message.casefold()
    if not re.search(
        r"\b(?:wtop|wtapiaj|zanikaj|zastosuj|wykonaj|namaluj|zast[aą]p|zmie[nń]|popraw|wyd[lł]u[zż])\w*",
        msg,
    ):
        return None
    if not (re.search(r"gradient", msg) and re.search(r"mas[kc]", msg)):
        return None
    if not re.search(r"layer\s*3", msg):
        return None
    if re.search(
        r"\bnie\s+(?:wykonuj|stosuj|maluj|nak[lł]adaj)\b[^.!?;\n]{0,80}gradient", msg
    ):
        return None
    # Zlecenie z koordynatami typu „od ... (1250, 2100) do ... (1250, 2550)”.
    points = re.search(
        r"od[^.!?;\n]{0,120}?\(\s*(\d{1,5})\s*,\s*(\d{1,5})\s*\)"
        r"[^.!?;\n]{0,100}?do[^.!?;\n]{0,80}?\(\s*(\d{1,5})\s*,\s*(\d{1,5})\s*\)",
        msg,
        re.S,
    )
    if points:
        return {
            "kind": "points",
            "start": (int(points[1]), int(points[2])),
            "end": (int(points[3]), int(points[4])),
        }
    percent = re.search(r"od\s*(\d{1,3})\s*%\s*do\s*(\d{1,3})\s*%", msg)
    if percent:
        return {"kind": "percent", "start": int(percent[1]), "end": int(percent[2])}
    return {"kind": "missing"}


async def run_lower_edge_fade(session, message, fade):
    if fade["kind"] == "missing":
        return (
            "Nie wykonano zmian. Podaj współrzędne, np. "
            "'Wykonaj gradient na masce Layer 3 od (1250, 2100) do (1250, 2550)', "
            "albo przedział 'od 60% do 73% wysokości dokumentu'."
        )

    async def inspect():
        return json.loads(
            extract_text(await session.call_tool("doc_inspect", arguments={}))
        )

    before = await inspect()
    layers = find_layers(before)
    width, height = before.get("width"), before.get("height")
    if (
        not before.get("name")
        or not isinstance(layers, list)
        or type(width) is not int
        or type(height) is not int
        or width < 10
        or height < 10
    ):
        return "Odmowa: nie potwierdzono aktywnego dokumentu i jego wymiarów."
    selected = [x for x in layers if isinstance(x, dict) and x.get("selected") is True]
    if len(selected) != 1 or str(selected[0].get("id")) != str(
        before.get("activeLayer")
    ):
        return "Odmowa: zaznacz dokładnie jedną warstwę, Layer 3 z muzykami."
    layer = selected[0]
    if (
        layer.get("name", "").casefold() != "layer 3"
        or layer.get("kind") != "Pixel"
        or layer.get("isBackground") is True
        or layer.get("locked") is True
        or layer.get("hasMask") is not True
    ):
        return "Odmowa: zaznacz odblokowaną warstwę Pixel Layer 3 z istniejącą maską."
    if re.search(r"\bid\s*21\b", message, re.I) and str(layer.get("id")) != "21":
        return "Odmowa: wskazano ID 21, ale Layer 3 ma inne ID."
    ids = [str(x.get("id")) for x in layers if isinstance(x, dict)]
    if len(ids) != len(layers) or len(set(ids)) != len(ids):
        return "Odmowa: niejednoznaczna lista warstw."
    if fade["kind"] == "percent":
        start_pct, end_pct = fade["start"], fade["end"]
        if not (0 <= start_pct < end_pct <= 100 and end_pct - start_pct >= 4):
            return "Nie wykonano zmian: zakres procentowy musi rosnąć i mieć minimum 4 punkty procentowe."
        start = [width // 2, round((height - 1) * start_pct / 100)]
        end = [width // 2, round((height - 1) * end_pct / 100)]
    else:
        start, end = list(fade["start"]), list(fade["end"])
    x1, y1 = start
    x2, y2 = end
    if not (
        0 <= x1 < width
        and 0 <= x2 < width
        and 0 <= y1 < y2 < height
        and abs(x1 - x2) <= 2
        and y2 - y1 >= 50
    ):
        return (
            "Nie wykonano zmian. Potrzebny pionowy gradient w granicach dokumentu, "
            "od wyższego punktu do niższego, minimum 50 pikseli długości."
        )
    bounds = layer.get("bounds")
    if isinstance(bounds, list) and len(bounds) == 4:
        # Maksymalnie zachowawczo: nie uruchamiaj gradientu całkowicie poza postaciami.
        if y2 <= bounds[1] or y1 >= bounds[1] + bounds[3]:
            return "Nie wykonano zmian: gradient nie przecina obszaru muzyków."
    # Weryfikacja przed zmianą jest skończona. To jedyne polecenie zapisu.
    response = extract_text(
        await session.call_tool(
            "command_run",
            arguments={
                "id": "paint.gradient",
                "params": {
                    "from": start,
                    "to": end,
                    "style": "linear",
                    "stops": [[0, "#ffffff"], [1, "#000000"]],
                    "transparency": [[0, 100], [1, 100]],
                    "reverse": False,
                    "dither": True,
                    "opacity": 100,
                    "mode": "normal",
                    "target": "mask",
                },
            },
        )
    )
    after = await inspect()
    now = find_layers(after)
    target = (
        next((l for l in now if str(l.get("id")) == str(layer.get("id"))), None)
        if isinstance(now, list)
        else None
    )
    if (
        after.get("name") != before.get("name")
        or not isinstance(now, list)
        or [str(x.get("id")) for x in now] != ids
        or not target
        or target.get("hasMask") is not True
    ):
        return (
            "UWAGA: wysłano paint.gradient, ale stan dokumentu wymaga ręcznej kontroli."
        )
    did_change = after.get("revision") != before.get("revision") or after.get(
        "history"
    ) != before.get("history")
    status = (
        "PhotoCraft zarejestrował zmianę dokumentu."
        if did_change
        else "Nie potwierdzono zmiany dokumentu — sprawdź wizualnie, czy gradient się pojawił."
    )
    return (
        f"Wysłano paint.gradient na istniejącej masce '{layer['name']}' "
        f"(ID {layer['id']}): biały {tuple(start)} → czarny {tuple(end)}. "
        f"{status} Nie dodano warstw ani nie zapisano pliku. "
        "doc_inspect nie potwierdza zawartości pikseli maski. "
        f"MCP: {response[:120]}"
    )


# --- SAM3: jawne polecenie wycinania na podstawie opisu (nie BiRefNet) ---
def sam3_prompt_from_message(message):
    """Zwraca doslowny opis obiektu lub None. Nie zgaduje zamiaru przy pytaniach.

    Format zalecany: 'Wytnij obiekt: blonde woman + phone'.
    Operator '+' oznacza dwa oddzielne przebiegi SAM3, nie pojedynczy opis.
    """
    if (
        not isinstance(message, str)
        or global_no_edit(message)
        or is_catalog_lookup(message)
    ):
        return None
    msg = message.strip()
    if not msg or len(msg) > 700:
        return None
    if re.match(
        r"^(?:czy|jak|dlaczego|po\s+co|możesz|mozesz|mógłbyś|moglbys)\b", msg, re.I
    ):
        return None
    if re.search(
        r"\b(?:nie|bez)\s+(?:wycinaj|wytnij|oddzielaj|wyodrębniaj|uruchamiaj)\b",
        msg,
        re.I,
    ):
        return None
    explicit = re.fullmatch(
        r"(?:wytnij|wyodrębnij|odseparuj)\s+(?:obiekt\s*:\s*|z\s+opisu\s*:\s*)(.+?)\s*[.!]?",
        msg,
        re.I | re.S,
    )
    if explicit:
        prompt = explicit.group(1).strip()
    else:
        # Tryb naturalny: tylko bezposredni rozkaz; nie przechwytuj narzedzi
        # BiRefNet, polecen pracy z maskami ani wypowiedzi wieloetapowych.
        match = re.fullmatch(r"(?:wytnij|wyodrębnij)\s+(.+?)\s*[.!]?", msg, re.I | re.S)
        if not match:
            return None
        prompt = match.group(1).strip()
        if prompt.casefold().startswith(("obiekt:", "z opisu:")):
            return None
        if re.search(
            r"\b(?:zaznaczon\w*|zaznaczeni\w*|tło|tła|tla|birefnet|"
            r"mask[ęaiy]?|warstw\w*|wklej|eksportuj|zapisz|potem|następnie)\b",
            prompt,
            re.I,
        ):
            return None
    if not prompt or len(prompt) > 400 or not re.search(r"\w", prompt, re.U):
        return None
    if re.search(
        r"\b(?:tło|tła|tla|birefnet|zaznaczon\w*|zaznaczeni\w*)\b", prompt, re.I
    ):
        return None
    if re.search(r"\b(?:potem|następnie|zapisz|eksportuj|wklej)\b", prompt, re.I):
        return None
    # Maksymalnie dwa niezalezne obiekty - nie zakladamy, ze model rozumie
    # relacje miedzy osoba i przedmiotem; backend ma je przetworzyc osobno.
    items = [x.strip() for x in prompt.split("+")]
    if len(items) > 2 or any(not part or len(part) > 200 for part in items):
        return None
    return " + ".join(items)


async def run_sam3_description(description):
    """Uruchamia osobny most. Bez potwierdzenia nie deklaruje sukcesu.

    Kontrakt mostu: --prompt OPIS, exit code 0 i w stdout linia
    'GOTOWE: SAM3 MASKA WARSTWY' lub 'GOTOWE: SAM3 ZAZNACZENIE'.
    """
    if not SAM3_SCRIPT.is_file():
        return (
            "SAM3: rozpoznano polecenie, ale nie ma jeszcze mostu ComfyUI → PhotoCraft. "
            "Nie wykonano zmian. Potrzebny jest plik "
            + str(SAM3_SCRIPT)
            + ". Najpierw wyeksportujemy działający workflow ComfyUI w formacie API."
        )
    # Windows: dziecko często domyślnie pisze w cp1250/cp852, a Gradio czyta UTF-8.
    # Wymuszamy UTF-8 po OBU stronach przechwyconego stdout/stderr.
    child_env = os.environ.copy()
    child_env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    try:
        finished = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, str(SAM3_SCRIPT), "--prompt", description],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            timeout=1200,
        )
    except subprocess.TimeoutExpired:
        return (
            "SAM3: przekroczono limit czasu. Sprawdź ComfyUI i PhotoCraft; "
            "część operacji mogła się wykonać. Nie uruchamiaj ponownie bez kontroli."
        )
    except OSError as exc:
        return f"SAM3: nie udało się uruchomić mostu: {exc}"
    output = (finished.stdout or "").strip()
    error = (finished.stderr or "").strip()
    if finished.returncode != 0:
        return (
            "SAM3: proces nie zakończył się pomyślnie. Sprawdź PhotoCraft, "
            "bo część operacji mogła się wykonać.\n\n" + (output + "\n" + error)[-3500:]
        )
    if "GOTOWE: SAM3 MASKA WARSTWY" in output:
        return (
            "SAM3: utworzono maskę warstwy na KOPII pełnego zdjęcia. "
            "Możesz poprawiać krawędzie białym i czarnym pędzlem. "
            "Aby zobaczyć przezroczystość, ukryj dolną warstwę Background. "
            "Nie zapisano projektu automatycznie.\n" + output[-3000:]
        )
    if "GOTOWE: SAM3 ZAZNACZENIE" in output:
        return (
            "SAM3: zaznaczenie gotowe, ale PhotoCraft nie potwierdził "
            "automatycznego utworzenia maski. OSTATNI KROK RĘCZNY: "
            "Warstwa → Maska warstwy → Pokaż zaznaczenie. "
            "Nie zapisano projektu.\n" + output[-3000:]
        )
    return (
        "SAM3: zakończono proces bez potwierdzenia maski lub zaznaczenia. "
        "Sprawdź dokumenty i warstwy; nie zakładam sukcesu.\n" + output[-3000:]
    )


def wants_birefnet_selection(message):
    """Wyraźne żądanie wycięcia *bieżącego zaznaczenia*, nie całego zdjęcia.

    Reguła deterministyczna — bez decyzji LLM i bez edycji przy pytaniach.
    """
    if global_no_edit(message) or is_catalog_lookup(message):
        return False
    msg = message.casefold().strip()
    if re.search(r"^(?:czy|jak|dlaczego|po\s+co|możesz|mozesz|mógłbyś|moglbys)\b", msg):
        return False
    if re.search(
        r"\b(?:nie|bez)\s+(?:wycinaj|wytnij|usuwaj|usuń|uruchamiaj|oddzielaj)\b", msg
    ):
        return False
    # Żądanie, a nie ogólna wzmianka o zaznaczonej warstwie lub narzędziu.
    action = re.search(
        r"\b(?:wytnij|wycinaj|oddziel|odseparuj|usu[nń]|uruchom|zastosuj)\b", msg
    )
    selection = re.search(
        r"\b(?:zaznaczon\w*|zaznaczeni\w*|wybran\w*\s+(?:obszar\w*|fragment\w*))\b", msg
    )
    subject = re.search(
        r"\b(?:obiekt\w*|grup\w*|posta[cć]\w*|osob\w*|fragment\w*|obszar\w*|t[lł]\w*|birefnet)\b",
        msg,
    )
    natural = re.search(r"\b(?:to|tego)\s*,?\s*co\s+(?:mam\s+)?zaznaczon\w*\b", msg)
    return bool(action and ((selection and subject) or natural))


async def run_birefnet_selection():
    """Uruchom zweryfikowany skrypt zaznaczenia poza sesją MCP Gradio."""
    if not BIREFNET_SELECTION_SCRIPT.is_file():
        return (
            f"Nie znaleziono skryptu zaznaczenia: {BIREFNET_SELECTION_SCRIPT}. "
            "Nie wykonano zmian. Skopiuj go do folderu comfy_bridge."
        )
    try:
        done = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, str(BIREFNET_SELECTION_SCRIPT)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return (
            "Wycinanie zaznaczenia przekroczyło limit czasu. "
            "Sprawdź ComfyUI i dokumenty PhotoCraft; część kroków mogła się wykonać."
        )
    except OSError as exc:
        return f"Nie udało się uruchomić wycinania zaznaczenia: {exc}"

    output = (done.stdout or "").strip()
    error = (done.stderr or "").strip()
    if done.returncode != 0:
        details = (output + "\n" + error)[-3500:]
        if "Brak aktywnego zaznaczenia" in details:
            return "Nie wykryto zaznaczenia. Zaznacz obiekt lub całą grupę prostokątem w PhotoCraft i spróbuj ponownie."
        return (
            "Nie ukończono wycinania zaznaczenia. Część kroków mogła się udać — "
            "sprawdź dokumenty i warstwy PhotoCraft.\n\n" + details
        )
    if "GOTOWE: nowa warstwa" not in output:
        return (
            "Skrypt zakończył się, ale nie potwierdził dodania nowej warstwy. "
            "Sprawdź PhotoCraft.\n" + output[-2500:]
        )
    summary = [
        line.strip()
        for line in output.splitlines()
        if any(
            key in line
            for key in (
                "Zaznaczenie:",
                "GOTOWE:",
                "Wyciecie:",
                "Maska:",
                "Nie zapisano",
                "Sprawdz czy",
            )
        )
    ]
    return (
        "Gotowe — BiRefNet przetworzył zaznaczony obszar i dodał nową warstwę.\n"
        + "\n".join(summary)
        + "\nOryginał nie został zapisany; pomocniczy dokument PNG pozostaje otwarty. "
        "Sprawdź krawędzie i położenie. Zaznaczenie ogranicza obszar, ale nie wybiera semantycznie jednej osoby."
    )


def wants_birefnet_cutout(message):
    """Wyłącznie jawne polecenie usunięcia tła z aktywnego dokumentu."""
    if global_no_edit(message) or is_catalog_lookup(message):
        return False
    msg = message.casefold()
    if re.search(
        r"\b(?:nie|bez)\s+(?:wycinaj|wytnij|usuwaj|usuń|zastosuj|uruchamiaj)\b", msg
    ):
        return False
    # Unikamy uruchomienia przy pytaniu typu 'czy możesz wyciąć tło?'.
    if re.search(r"^\s*(?:czy|jak|dlaczego|po\s+co|możesz|mógłbyś|mozesz)\b", msg):
        return False
    action = re.search(
        r"\b(?:wytnij|wycinaj|usuń|usun|usu[nń]|oddziel|odseparuj|uruchom|zastosuj)\b",
        msg,
    )
    subject = re.search(r"\b(?:tło|tla|tła|birefnet)\b", msg)
    return bool(action and subject)


async def run_birefnet_cutout():
    if not BIREFNET_SCRIPT.is_file():
        return f"Nie znaleziono mostu BiRefNet: {BIREFNET_SCRIPT}. Nie wykonano zmian."
    try:
        done = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, str(BIREFNET_SCRIPT)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return (
            "BiRefNet przekroczył limit czasu. Sprawdź ComfyUI i dokumenty PhotoCraft."
        )
    except OSError as exc:
        return f"Nie udało się uruchomić mostu BiRefNet: {exc}"
    output = (done.stdout or "").strip()
    error = (done.stderr or "").strip()
    # Starsza wersja mostu raportowała SystemExit(0) jako BLAD i zwracała kod 1.
    # Akceptujemy ten konkretny przypadek wyłącznie przy potwierdzonej nowej warstwie.
    legacy_success = (
        done.returncode == 1
        and "GOTOWE: nowa warstwa" in output
        and "SystemExit: 0" in error
        and "BLAD: 0" in error
    )
    if done.returncode != 0 and not legacy_success:
        return (
            "BiRefNet nie ukończył operacji. Możliwe, że część kroków została wykonana; "
            "sprawdź zakładki i warstwy PhotoCraft.\n\n"
            + (output + "\n" + error)[-3500:]
        )
    if "GOTOWE: nowa warstwa" not in output:
        return (
            "Proces zakończył się bez potwierdzenia nowej warstwy. Sprawdź PhotoCraft.\n"
            + output[-2500:]
        )
    summary = [
        line.strip()
        for line in output.splitlines()
        if any(
            key in line
            for key in ("GOTOWE:", "Wyciecie:", "Maska:", "Nie zapisano", "Sprawdz czy")
        )
    ]
    return (
        "BiRefNet zakończył wycinanie tła i dodał nową warstwę.\n"
        + "\n".join(summary)
        + "\nOryginał nie został zapisany. Sprawdź pozycję warstwy; pomocniczy PNG pozostał otwarty."
    )


async def process_message(message, history):
    # Gdy występuje globalny zakaz, nie udostępniaj narzędzi edycyjnych.
    # Pytania katalogowe i odczyty są obsługiwane dalej bez zmian.
    if global_no_edit(message) and not is_catalog_lookup(message):
        if re.search(
            r"doc_inspect|activeLayer|lista warstw|stan dokumentu|identyfikator|zaznaczon",
            message,
            re.I,
        ):
            pass  # Późniejsza bezpośrednia ścieżka odczytu.
        elif re.search(r"obejrzyj|zobacz|oce[nń]|analiz", message, re.I):
            pass  # Analiza bez edycji.
        else:
            return "Nie wykonano zmian: polecenie zawiera globalny zakaz edycji."
    # Oba mosty otwierają własne sesje MCP; wybór zaznaczenia ma pierwszeństwo
    # przed ogólnym "wytnij tło". Nie wysyłamy tych działań do LLM.
    if wants_birefnet_selection(message):
        print("WYBRANE NARZĘDZIA: ['birefnet_selection']", flush=True)
        return await run_birefnet_selection()
    # BiRefNet uruchamiamy jako osobny proces PRZED otwarciem sesji MCP,
    # bo skrypt mostu zarządza własną sesją i przełączaniem dokumentów.
    if wants_birefnet_cutout(message):
        print("WYBRANE NARZĘDZIA: ['birefnet_cutout']", flush=True)
        return await run_birefnet_cutout()

    # SAM3: tekst opisowy (zanim wiadomość trafi do LLM).
    sam3_description = sam3_prompt_from_message(message)
    if sam3_description is not None:
        print("WYBRANE NARZĘDZIA: ['sam3_description']", flush=True)
        return await run_sam3_description(sam3_description)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        if item.get("role") in ("user", "assistant"):
            content = item.get("content")
            if isinstance(content, str):
                messages.append({"role": item["role"], "content": content})
    messages.append({"role": "user", "content": message})

    # Tylko zapytania katalogowe: nie wysyłamy ich do modelu ani MCP.
    # Wyszukiwanie korzysta ze słów kluczowych wyodrębnionych z pytania.
    if is_catalog_lookup(message):
        print(
            "WYBRANE NARZĘDZIA: ['photocraft_command_search'] (katalog bezpośredni)",
            flush=True,
        )
        try:
            queries = catalog_search_queries(message)
            results = [
                f"Zapytanie katalogowe: {query}\n{search_command_catalog(query)}"
                for query in queries
            ]
            return "\n\n".join(results) + "\n\nNie wykonano żadnych zmian."
        except Exception as error:
            return f"Nie udało się odczytać katalogu: {error}"

    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # v3.3: płynne wtopienie dolnej krawędzi, tylko na istniejącej masce.
            fade = parse_lower_edge_fade(message)
            if fade is not None:
                print("WYBRANE NARZĘDZIA: ['lower_edge_fade_mask']", flush=True)
                return await run_lower_edge_fade(session, message, fade)
            # v3.2: biała maska na wskazanej, zaznaczonej warstwie obrazu.
            if wants_reveal_all_on_image(message):
                print("WYBRANE NARZĘDZIA: ['reveal_all_on_image']", flush=True)
                return await run_reveal_all_on_image(session, message)
            # v3.1: przygotowanie czarnej maski do ręcznego malowania — bez punktów X/Y.
            manual_brightness = manual_face_mask_request(message)
            if manual_brightness is not None:
                print("WYBRANE NARZĘDZIA: ['prepare_manual_face_mask']", flush=True)
                return await prepare_manual_face_mask(
                    session, message, manual_brightness
                )
            # v3.0: lokalne doświetlenie z jawnymi współrzędnymi, bez zgadywania twarzy.
            face_args = face_light_request(message)
            if face_args is not None:
                print("WYBRANE NARZĘDZIA: ['face_light_masked']", flush=True)
                return await run_face_light(session, message, face_args)
            # v2.8: Gradient Map + clipping, zanim pytanie trafi do modelu lub odczytu.
            if wants_santana_gradient_clipped(message):
                print(
                    "WYBRANE NARZĘDZIA: ['new_gradient_map_plus_clipping']", flush=True
                )
                return await run_santana_gradient_clipped(session, message)
            # Dwuetapowy zapis: musi poprzedzać odczyt i jednofunkcyjny clipping.
            if wants_new_bw_clipped(message):
                print(
                    "WYBRANE NARZĘDZIA: ['new_bw_plus_clipping'] (2-etapowo)",
                    flush=True,
                )
                return await run_new_bw_clipped(session, message)
            # Jawne pytanie o warstwy: deterministyczny odczyt, bez zgadywania przez LLM.
            read_request = bool(
                re.search(
                    r"doc_inspect|activeLayer|identyfikator|identyfikatory|lista warstw|stan (?:aktywnego )?dokumentu|zaznaczon(?:ej|a) warstw|sprawd[zź]\s+(?:aktywny\s+)?dokument",
                    message,
                    re.I,
                )
            )
            write_request = bool(
                re.search(
                    r"\b(?:przesuń|przesun|zmień|zmien|usuń|usun|dodaj|wstaw|przeskaluj|utwórz|utworz|skaluj|zwolnij|zwolnic|zwolnić|odłącz|odlacz|release)\b",
                    message,
                    re.I,
                )
            )
            catalog_request = bool(
                re.search(
                    r"katalog|polecen|komend|command_list|black\s*&\s*white|blackwhite|czarno.bia",
                    message,
                    re.I,
                )
            )
            if read_request and not write_request and not catalog_request:
                print(
                    "WYBRANE NARZĘDZIA: ['photocraft_doc_inspect'] (odczyt bezpośredni)",
                    flush=True,
                )
                data = json.loads(
                    extract_text(await session.call_tool("doc_inspect", arguments={}))
                )
                layers = find_layers(data)
                if not isinstance(layers, list):
                    return (
                        "Nie udało się odczytać listy warstw z doc_inspect.\n"
                        + json.dumps(data, ensure_ascii=False)[:2500]
                    )
                lines = [
                    f"Dokument: {data.get('name', '(brak nazwy)')}",
                    f"activeLayer: {data.get('activeLayer')}",
                    "Warstwy:",
                ]
                for layer in layers:
                    if isinstance(layer, dict):
                        lines.append(
                            f"- ID {layer.get('id')}: {layer.get('name')}"
                            + (" [zaznaczona]" if layer.get("selected") is True else "")
                            + f" | rodzaj={layer.get('kind')} | maska={layer.get('hasMask', 'brak danych')}"
                        )
                lines.append("Nie wykonano żadnych zmian.")
                return "\n".join(lines)
            # Jednoznaczne polecenia clipping obsługujemy bez zgadywania przez model.
            # Walidacja i blokada operacji pozostają w run_clipping_tool.
            selected_clipping = select_tool_group(message)
            if selected_clipping and selected_clipping[0]["function"]["name"] in (
                "photocraft_create_clipping_mask",
                "photocraft_release_clipping_mask",
            ):
                release = (
                    selected_clipping[0]["function"]["name"]
                    == "photocraft_release_clipping_mask"
                )
                print(
                    "WYBRANE NARZĘDZIA:",
                    [selected_clipping[0]["function"]["name"]],
                    "(bezpośrednio)",
                    flush=True,
                )
                return await run_clipping_tool(
                    session, {}, message, set(), release=release
                )

            available = await session.list_tools()
            tools = []
            for tool in available.tools:
                if tool.name in READ_TOOLS:
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": "photocraft_" + tool.name,
                                "description": tool.description or "",
                                "parameters": tool.inputSchema,
                            },
                        }
                    )
            # Wybieraj tylko narzędzia pasujące do aktualnej intencji.
            # Nie interpretuj samego słowa "gradient" jako polecenia jego dodania.
            selected = select_tool_group(message)
            if global_no_edit(message):
                selected = [
                    tool
                    for tool in selected
                    if tool["function"]["name"]
                    in ("photocraft_command_search", "photocraft_analyze_photo")
                ]
                tools = [
                    tool
                    for tool in tools
                    if tool["function"]["name"]
                    in ("photocraft_session_list", "photocraft_doc_inspect")
                ]
            # Przy jednoznacznym poleceniu edycji pokazuj TYLKO wybraną
            # funkcję. Wcześniejszy test z jednym narzędziem działał poprawnie.
            # Przy odczycie udostępniaj wyłącznie narzędzia odczytu.
            if selected:
                tools = selected
            elif re.search(
                r"(?:dodaj|utw[oó]rz|stw[oó]rz)\s+nowy\s+dokument", message, re.I
            ):
                for tool in available.tools:
                    if tool.name == "doc_new":
                        tools = [
                            {
                                "type": "function",
                                "function": {
                                    "name": "photocraft_doc_new",
                                    "description": tool.description or "",
                                    "parameters": tool.inputSchema,
                                },
                            }
                        ]
                        break
            print(
                "WYBRANE NARZĘDZIA:", [t["function"]["name"] for t in tools], flush=True
            )

            used_actions = set()
            tool_results = []

            for _ in range(7):
                # Po wyniku narzędzia wyłączamy rozumowanie, aby model
                # wygenerował odpowiedź zamiast pustego content.
                answer = await asyncio.to_thread(
                    ollama_chat, messages, tools, not bool(tool_results)
                )
                calls = answer.get("tool_calls") or []
                if not calls:
                    content = (answer.get("content") or "").strip()
                    if content:
                        return content
                    if tool_results:
                        # Wynik MCP/katalogu jest ważniejszy niż pusta odpowiedź LLM.
                        return "\n\n".join(tool_results[-3:])
                    return "Model nie wywołał narzędzia ani nie zwrócił odpowiedzi. Nie wprowadzono zmian. Spróbuj ponownie."
                messages.append(answer)

                for call in calls:
                    name = call["function"]["name"]
                    arguments = call["function"].get("arguments", {})
                    print("Wywołanie:", name, arguments, flush=True)
                    try:
                        if name == "photocraft_create_clipping_mask":
                            result_text = await run_clipping_tool(
                                session, arguments, message, used_actions
                            )
                        elif name == "photocraft_release_clipping_mask":
                            result_text = await run_clipping_tool(
                                session, arguments, message, used_actions, release=True
                            )
                        elif name == "photocraft_command_search":
                            result_text = search_command_catalog(
                                arguments.get("query", "")
                            )
                        elif name in {
                            "photocraft_image_place",
                            "photocraft_layer_translate",
                            "photocraft_layer_scale",
                            "photocraft_layer_reorder",
                        }:
                            result_text = await run_image_tool(
                                session, name, arguments, message, used_actions
                            )
                        elif name in (
                            "photocraft_layer_add",
                            "photocraft_layer_delete",
                        ):
                            result_text = await run_layer_tool(
                                session, name, arguments, message, used_actions
                            )
                        elif name in ("photocraft_shape_add", "photocraft_text_add"):
                            result_text = await run_draw_tool(
                                session, name, arguments, message, used_actions
                            )
                        elif name == "photocraft_analyze_photo":
                            result_text = await analyze_photo(
                                session, message, used_actions
                            )
                        elif name == "photocraft_contrast_existing":
                            result_text = await run_existing_contrast_tool(
                                session, arguments, message, used_actions
                            )
                        elif name == "photocraft_mask_protect":
                            result_text = await run_mask_protect_tool(
                                session, arguments, message, used_actions
                            )
                        elif name == "photocraft_gradient_adjustment":
                            result_text = await run_gradient_adjustment_tool(
                                session, arguments, message, used_actions
                            )
                        elif name == "photocraft_levels_existing":
                            result_text = await run_existing_levels_tool(
                                session, arguments, message, used_actions
                            )
                        elif name == "photocraft_levels_new":
                            result_text = await run_levels_tool(
                                session, arguments, message, used_actions
                            )
                        elif name == "photocraft_brightness_contrast":
                            result_text = await run_adjustment_tool(
                                session, arguments, message, used_actions
                            )
                        elif name.startswith("photocraft_") and name[11:] in READ_TOOLS:
                            mcp_name = name[11:]
                            # Zachowujemy istniejącą funkcję tworzenia dokumentów.
                            result = await session.call_tool(
                                mcp_name, arguments=arguments
                            )
                            result_text = extract_text(result)
                        else:
                            result_text = "Narzędzie zablokowane."
                    except Exception as error:
                        result_text = "Błąd MCP: " + str(error)
                        print("SZCZEGÓŁY BŁĘDU MCP:", repr(error), flush=True)
                    print("Wynik:", result_text[:2000], flush=True)
                    if name == "photocraft_analyze_photo":
                        return result_text

                    tool_results.append(result_text)
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": result_text}
                    )
            return "Przekroczono limit wywołań narzędzi."


def odpowiedz(message, history):
    try:
        return asyncio.run(process_message(message, history))
    except Exception as error:
        traceback.print_exception(error)
        return f"Błąd połączenia: {error}"


# --- PANEL PhotoCraft AI: kontrolki obok zachowanego czatu Qwen ---
# Wszystkie przyciski korzystają z tych samych sprawdzonych funkcji,
# co dotychczasowy czat. Nie łączymy dwóch sesji MCP równocześnie.

SAM3_RESULTS_DIR = BASE_DIR / "comfy_bridge" / "wyniki_sam3"


def panel_birefnet_full():
    return odpowiedz("Wytnij tło", [])


def panel_birefnet_selection():
    return odpowiedz("Wytnij zaznaczony obiekt", [])


def panel_sam3(first, second):
    """Obsługiwany kontrakt: maksymalnie dwa opisy łączone operatorem +."""
    one = (first or "").strip()
    two = (second or "").strip()
    if not one:
        return "Podaj pierwszy obiekt. Nie uruchomiono SAM3."
    if any(
        not 1 <= len(item) <= 200 or "+" in item or "\n" in item or "\r" in item
        for item in (one, two)
        if item
    ):
        return "Opisy muszą mieć do 200 znaków; znak + i nowe linie są niedozwolone w polach."
    description = " + ".join([item for item in (one, two) if item])
    # Ta sama ścieżka co w Gradio dla komendy tekstowej, z pełnym
    # potwierdzeniem maski i bez modyfikacji dokumentu źródłowego.
    return odpowiedz("Wytnij obiekt: " + description, [])


async def _panel_analyze_photo():
    # Używa istniejącej funkcji Qwen3-VL, bez dodawania lub modyfikacji warstw.
    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await analyze_photo(session, "Oceń aktualne zdjęcie", set())


def _panel_root_error(error):
    """Pokaż rzeczywisty błąd spod ExceptionGroup (MCP/AnyIO)."""
    current = error
    for _ in range(8):
        children = getattr(current, "exceptions", None)
        if not children:
            break
        current = children[0]
    return str(current) or type(current).__name__


def panel_analyze_photo():
    try:
        return asyncio.run(_panel_analyze_photo())
    except Exception as error:
        traceback.print_exception(error)
        return (
            "Nie udało się przeanalizować fotografii: "
            + _panel_root_error(error)
            + "\nNie wprowadzono korekt w PhotoCraft."
        )


def panel_open_results():
    """Otwiera wyłącznie stały folder wyników SAM3 na komputerze Windows."""
    if not SAM3_RESULTS_DIR.is_dir():
        return "Folder wyników jeszcze nie istnieje: " + str(SAM3_RESULTS_DIR)
    try:
        os.startfile(str(SAM3_RESULTS_DIR))
        return "Otworzono folder wyników: " + str(SAM3_RESULTS_DIR)
    except (OSError, AttributeError) as exc:
        return (
            "Nie udało się otworzyć folderu: "
            + str(exc)
            + "\nŚcieżka: "
            + str(SAM3_RESULTS_DIR)
        )


# Rozwijana ściągawka — interfejs bez zmiany logiki asystenta.
COMMAND_GUIDE = """
### Polecenia czatu

| Cel | Polecenie | Informacja |
|---|---|---|
| Usunięcie tła | **Wytnij tło** | BiRefNet na całym zdjęciu. |
| Wycięcie zaznaczenia | **Wytnij zaznaczony obiekt** | Przedtem wykonaj zaznaczenie w PhotoCraft. |
| Wycięcie z opisu | **Wytnij obiekt: blonde woman + phone** | SAM3 łączy maksymalnie dwa opisy i tworzy edytowalną maskę. |
| Ocena fotografii | **Oceń aktualne zdjęcie** | Qwen3-VL; tylko analiza, bez zmian. |
| Warstwy | **Pokaż listę warstw** | Informacje z aktualnego dokumentu. |
| Korekta | **Dodaj nową warstwę Jasność/Kontrast: jasność 15, kontrast 10** | Wymaga jawnych wartości. |
| Poziomy | **Dodaj nową warstwę Poziomy: punkt czerni 10, gamma 1.1, punkt bieli 245** | Osobna, niedestrukcyjna korekta. |
| Katalog komend | **Pokaż komendy dotyczące gradientMap** | Bez wykonywania zmian. |

**Ważne:** czat Qwen pozostaje dostępny do innych zadań. SAM3 i BiRefNet
mogą otwierać pomocnicze dokumenty; oryginału nie zapisują automatycznie.
Zapis PSD i eksport końcowy wykonywany jest na razie ręcznie w PhotoCraft.
"""

with gr.Blocks(title="PhotoCraft AI") as demo:
    gr.Markdown(
        "# ✨ PhotoCraft AI\n**PhotoCraft · ComfyUI · SAM3 · BiRefNet · Qwen3-VL**"
    )
    gr.Markdown(
        "Wybierz gotowe narzędzie albo skorzystaj z pełnego czatu AI poniżej. "
        "Przed wycinaniem otwórz zdjęcie w PhotoCraft."
    )

    with gr.Tabs():
        with gr.Tab("🎯 SAM3 — obiekty z opisu"):
            gr.Markdown("### Wycinanie obiektów i tworzenie edytowalnej maski")
            gr.Markdown(
                "Wpisz nazwę obiektu (najpewniej po angielsku). "
                "Drugi opis jest opcjonalny; jeśli go dodasz, SAM3 połączy obie maski."
            )
            with gr.Row():
                sam3_first = gr.Textbox(
                    label="Obiekt 1", placeholder="np. blonde woman", lines=1
                )
                sam3_second = gr.Textbox(
                    label="Obiekt 2 (opcjonalnie)", placeholder="np. phone", lines=1
                )
            sam3_button = gr.Button("✨ Uruchom SAM3 i utwórz maskę", variant="primary")
            sam3_output = gr.Textbox(label="Wynik SAM3", lines=9, interactive=False)
            sam3_button.click(
                fn=panel_sam3,
                inputs=[sam3_first, sam3_second],
                outputs=sam3_output,
                concurrency_limit=1,
            )
            gr.Markdown(
                "**Efekt:** nowy dokument z pełnym RGB, kopią warstwy i maską do "
                "poprawiania pędzlem. Ukryj dolny `Background`, aby zobaczyć przezroczystość. "
                "Najwyżej dwa opisy w tej wersji."
            )

        with gr.Tab("✂️ BiRefNet — usuwanie tła"):
            gr.Markdown("### Szybkie wycinanie")
            with gr.Row():
                full_button = gr.Button(
                    "🖼️ Wytnij tło z całego zdjęcia", variant="primary"
                )
                selection_button = gr.Button("▣ Wytnij zaznaczony obiekt")
            birefnet_output = gr.Textbox(
                label="Wynik BiRefNet", lines=8, interactive=False
            )
            full_button.click(
                fn=panel_birefnet_full,
                inputs=None,
                outputs=birefnet_output,
                concurrency_limit=1,
            )
            selection_button.click(
                fn=panel_birefnet_selection,
                inputs=None,
                outputs=birefnet_output,
                concurrency_limit=1,
            )
            gr.Markdown(
                "W trybie **zaznaczonego obiektu** najpierw zaznacz obszar w PhotoCraft. "
                "Pozostaw trochę zapasu wokół postaci."
            )

        with gr.Tab("📷 Ocena fotografii"):
            gr.Markdown("### Analiza otwartego zdjęcia — Qwen3-VL")
            gr.Markdown(
                "Ocena kompozycji, światła, kolorów i ostrości. "
                "Asystent proponuje maksymalnie trzy poprawki, ale **nie edytuje zdjęcia**."
            )
            analyze_button = gr.Button("🔎 Oceń aktualną fotografię", variant="primary")
            analysis_output = gr.Textbox(
                label="Analiza i sugestie", lines=10, interactive=False
            )
            analyze_button.click(
                fn=panel_analyze_photo,
                inputs=None,
                outputs=analysis_output,
                concurrency_limit=1,
            )
            gr.Markdown(
                "Po analizie możesz omówić propozycje w czacie na dole. "
                "Korekty wykonujemy dopiero na Twoje wyraźne polecenie."
            )

        with gr.Tab("📂 Wyniki i eksport"):
            gr.Markdown("### Wyniki wycinania SAM3")
            gr.Markdown(f"`{SAM3_RESULTS_DIR}`")
            folder_button = gr.Button("📁 Otwórz folder wyników")
            folder_output = gr.Textbox(label="Folder", lines=3, interactive=False)
            folder_button.click(
                fn=panel_open_results, inputs=None, outputs=folder_output
            )
            gr.Markdown(
                "**Zapis PSD i PNG:** na tym etapie wykonuj go z menu PhotoCraft. "
                "Nie dodaję przycisków pozorujących działający eksport. "
                "SAM3 zapisuje też swoje pliki pomocnicze `cutout.png` i `mask.png`."
            )

    gr.Markdown("---\n## 💬 Czat PhotoCraft AI — pozostałe zadania")
    gr.Markdown(
        "Czat pozostaje dostępny niezależnie od narzędzi powyżej: "
        "analiza, korekty, warstwy, polecenia PhotoCraft i pytania do Qwen."
    )
    gr.ChatInterface(
        fn=odpowiedz,
        title="Asystent PhotoCraft AI",
        description="Qwen3 + Qwen3-VL + PhotoCraft MCP · SAM3 · BiRefNet",
    )
    with gr.Accordion("📋 Rozkład jazdy — przykładowe polecenia", open=False):
        gr.Markdown(COMMAND_GUIDE)

demo.queue(default_concurrency_limit=1)
demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
