"""SAM3: eksport -> ComfyUI -> nowy dokument -> maska edytowalna.

Po wczytaniu zaznaczenia próbuje wykonać Reveal Selection przez MCP.
Jeśli wywołanie nie jest obsługiwane, zachowuje zaznaczenie do ręcznej operacji.
Nie modyfikuje ani nie zapisuje źródłowego dokumentu.
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import traceback
import uuid
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from comfy_sam3_api_v1 import run as run_comfy_sam3

MCP_URL = "http://127.0.0.1:8766/mcp"

BRIDGE_DIR = Path(__file__).resolve().parent
BASE_DIR = BRIDGE_DIR.parent

EXPORT_DIR = Path(os.environ.get("PHOTOCRAFT_EXPORT_DIR", str(BASE_DIR / "exports")))
IMPORT_DIR = Path(os.environ.get("PHOTOCRAFT_IMPORT_DIR", str(BASE_DIR / "import")))
RESULT_DIR = BRIDGE_DIR / "wyniki_sam3"


def text(result):
    message = "\n".join(part.text for part in result.content if part.type == "text")
    if result.isError:
        raise RuntimeError("MCP: " + message)
    return message


def decode(result):
    return json.loads(text(result))


def doclist(data):
    if isinstance(data.get("documents"), list):
        return data["documents"]
    return (data.get("session") or {}).get("documents")


def active(data):
    val = data.get("active")
    if val is None:
        val = (data.get("session") or {}).get("active")
    return val


def stable_layers(inspection):
    values = inspection.get("layers")
    if not isinstance(values, list) or not values:
        raise RuntimeError("Brak listy warstw w doc_inspect.")
    ids = [str(x.get("id")) for x in values]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Niejednoznaczne identyfikatory warstw.")
    return values


async def attempt_reveal_selection(
    session, expected_name, expected_layer_id, expected_ids
):
    """Próba jednego polecenia Reveal Selection; fallback do menu GUI tylko przy błędzie.

    Sygnał sukcesu: doc_inspect potwierdza hasMask=True na dokładnie tej samej warstwie
    w dokładnie tym samym dokumencie i bez zmian listy warstw. Nie wolno nadpisywać
    wcześniej istniejącej maski. Nie dotykamy oryginalnego dokumentu.
    """

    async def inspect_checked():
        state = decode(await session.call_tool("doc_inspect", arguments={}))
        if state.get("name") != expected_name:
            raise RuntimeError("Aktywny dokument zmienił się podczas tworzenia maski.")
        current = stable_layers(state)
        if [str(x["id"]) for x in current] != expected_ids:
            raise RuntimeError("Układ warstw zmienił się podczas tworzenia maski.")
        if str(state.get("activeLayer")) != str(expected_layer_id):
            raise RuntimeError("Aktywna warstwa uległa zmianie; przerwano.")
        chosen = next(
            (x for x in current if str(x["id"]) == str(expected_layer_id)), None
        )
        if chosen is None:
            raise RuntimeError("Nie odnaleziono warstwy docelowej.")
        return state, chosen

    before, layer = await inspect_checked()
    if layer.get("hasMask") is True:
        # Nie nadpisujemy żadnej istniejącej maski, również jeżeli
        # wcześniejszy etap został ponownie wykonany.
        raise RuntimeError("Kopia ma już maskę; odmowa jej zastąpienia.")
    if layer.get("kind") != "Pixel" or layer.get("locked") is True:
        raise RuntimeError("Warstwa docelowa nie jest odblokowaną warstwą Pixel.")
    bounds = before.get("selectionBounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return False, "Brak potwierdzonego zaznaczenia — maska nie została utworzona."

    errors = []
    try:
        text(
            await session.call_tool(
                "command_run",
                arguments={
                    "id": "layer.layerMask.revealSelection",
                    "params": {"layer": expected_layer_id},
                },
            )
        )
    except Exception as exc:
        errors.append("command_run: " + str(exc))
    _, after_layer = await inspect_checked()
    if after_layer.get("hasMask") is True:
        return True, "command_run"

    # Nie każda instalacja dopuszcza command_run dla komendy oznaczonej
    # enabled=false; standardowa operacja menu może mieć osobną ścieżkę.
    # Próbujemy ją tylko raz i tylko jeśli maska wciąż NIE istnieje.
    try:
        text(
            await session.call_tool(
                "ui_menu_invoke", arguments={"id": "layer.layerMask.revealSelection"}
            )
        )
    except Exception as exc:
        errors.append("ui_menu_invoke: " + str(exc))
    _, after_menu_layer = await inspect_checked()
    if after_menu_layer.get("hasMask") is True:
        return True, "ui_menu_invoke"
    return False, (
        " / ".join(errors) if errors else "PhotoCraft nie potwierdził hasMask=True."
    )


async def main(description):
    if not EXPORT_DIR.is_dir():
        raise FileNotFoundError("Brak folderu eksportu PhotoCraft: " + str(EXPORT_DIR))
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    input_name = "pc_sam3_input_" + token + ".png"
    input_path = EXPORT_DIR / input_name
    print("1/5 Kontrola otwartego zdjęcia i eksport...", flush=True)
    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            exposed = {x.name for x in (await session.list_tools()).tools}
            required = {
                "session_list",
                "doc_inspect",
                "doc_export",
                "doc_open",
                "ui_menu_invoke",
                "command_run",
            }
            if not required.issubset(exposed):
                raise RuntimeError(
                    "Brakuje narzędzi MCP: " + ", ".join(sorted(required - exposed))
                )
            source = decode(await session.call_tool("doc_inspect", arguments={}))
            stable_layers(source)
            if not source.get("name") or not all(
                type(source.get(k)) is int for k in ("width", "height")
            ):
                raise RuntimeError(
                    "Aktywny dokument nie ma poprawnej nazwy lub wymiarów."
                )
            W, H = source["width"], source["height"]
            if W < 32 or H < 32:
                raise RuntimeError("Dokument jest zbyt mały do testu SAM3.")
            state = decode(await session.call_tool("session_list", arguments={}))
            original_index = active(state)
            docs = doclist(state)
            if (
                type(original_index) is not int
                or not isinstance(docs, list)
                or original_index >= len(docs)
            ):
                raise RuntimeError(
                    "Nie można ustalić aktywnego dokumentu; przerwano przed zmianami."
                )
            text(await session.call_tool("doc_export", arguments={"path": input_name}))
            if not input_path.is_file() or input_path.stat().st_size < 100:
                raise RuntimeError(
                    "Eksport PhotoCraft nie utworzył pliku: " + str(input_path)
                )
            with input_path.open("rb") as f:
                if f.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise RuntimeError("PhotoCraft wyeksportował format inny niż PNG.")
            print("2/5 SAM3: opis, maska i wycięcie...", flush=True)
            cutout, mask = await asyncio.to_thread(
                run_comfy_sam3, input_path, description, RESULT_DIR
            )
            # PNG z białym obiektem na czarnym tle z ComfyUI. Ładowanie kanału red
            # powinno przenieść miękkie wartości (jasność) do zaznaczenia docelowego.
            # Dla celów odtworzenia wyniku zostawiamy pliki na dysku.
            original_copy = IMPORT_DIR / ("pc_sam3_source_" + token + ".png")
            mask_copy = IMPORT_DIR / ("pc_sam3_mask_" + token + ".png")
            shutil.copy2(input_path, original_copy)
            shutil.copy2(mask, mask_copy)
            # Nie działaj dalej, jeśli użytkownik przełączył dokument podczas SAM3.
            check = decode(await session.call_tool("doc_inspect", arguments={}))
            if (check.get("name"), check.get("width"), check.get("height")) != (
                source["name"],
                W,
                H,
            ):
                raise RuntimeError(
                    "Aktywny dokument się zmienił podczas SAM3. Wyniki zapisane: "
                    + str(cutout)
                )
            print(
                "3/5 Otwieram pełne zdjęcie jako NOWY dokument (oryginał nietknięty)...",
                flush=True,
            )
            text(
                await session.call_tool(
                    "doc_open", arguments={"path": str(original_copy)}
                )
            )
            s1 = decode(await session.call_tool("session_list", arguments={}))
            copy_index = active(s1)
            if (
                not isinstance(doclist(s1), list)
                or len(doclist(s1)) != len(docs) + 1
                or type(copy_index) is not int
                or copy_index == original_index
            ):
                raise RuntimeError("Nie potwierdzono nowej zakładki pełnego zdjęcia.")
            photo = decode(await session.call_tool("doc_inspect", arguments={}))
            old_layers = stable_layers(photo)
            if (photo.get("width"), photo.get("height")) != (W, H):
                raise RuntimeError("Kopia pełnego zdjęcia ma inne wymiary.")
            # Po otwarciu PNG zwykle jest aktywna warstwa Background.
            selected = [x for x in old_layers if x.get("selected") is True]
            if len(selected) != 1 or str(selected[0].get("id")) != str(
                photo.get("activeLayer")
            ):
                raise RuntimeError(
                    "W kopii zdjęcia nie potwierdzono aktywnej warstwy; nie duplikuję."
                )
            if selected[0].get("kind") not in ("Pixel", None):
                raise RuntimeError("Kopia nie zawiera wybranej warstwy obrazu Pixel.")
            text(
                await session.call_tool(
                    "command_run",
                    arguments={
                        "id": "layer.duplicate",
                        "params": {"layer": selected[0]["id"]},
                    },
                )
            )
            photo2 = decode(await session.call_tool("doc_inspect", arguments={}))
            layers2 = stable_layers(photo2)
            old_ids = {str(x["id"]) for x in old_layers}
            new = [x for x in layers2 if str(x["id"]) not in old_ids]
            if len(new) != 1 or len(layers2) != len(old_layers) + 1:
                raise RuntimeError(
                    "Nie potwierdzono kopii pełnej warstwy. Przerwano przed maską."
                )
            copy_layer = new[0]
            if str(photo2.get("activeLayer")) != str(copy_layer["id"]):
                raise RuntimeError(
                    "Nowa warstwa nie jest aktywna — nie chcę maskować niewłaściwej."
                )
            print(
                f"   Kopia zdjęcia: warstwa {copy_layer.get('name')} ID {copy_layer['id']}",
                flush=True,
            )
            print("4/5 Otwieram czarno-białą maskę SAM3...", flush=True)
            text(
                await session.call_tool("doc_open", arguments={"path": str(mask_copy)})
            )
            s2 = decode(await session.call_tool("session_list", arguments={}))
            mask_index = active(s2)
            if (
                not isinstance(doclist(s2), list)
                or len(doclist(s2)) != len(docs) + 2
                or type(mask_index) is not int
                or mask_index in (original_index, copy_index)
            ):
                raise RuntimeError("Nie potwierdzono osobnej zakładki maski.")
            mask_inspection = decode(
                await session.call_tool("doc_inspect", arguments={})
            )
            if (mask_inspection.get("width"), mask_inspection.get("height")) != (W, H):
                raise RuntimeError(
                    "Maska ma inne wymiary niż źródło — nie wczytuję zaznaczenia."
                )
            print("5/5 Przygotowuję zaznaczenie na kopii pełnego obrazu...", flush=True)
            text(
                await session.call_tool(
                    "ui_menu_invoke", arguments={"id": f"window.document.{copy_index}"}
                )
            )
            ready = decode(await session.call_tool("doc_inspect", arguments={}))
            if (ready.get("name"), ready.get("width"), ready.get("height")) != (
                photo2.get("name"),
                W,
                H,
            ):
                raise RuntimeError("Nie potwierdzono przełączenia na kopię zdjęcia.")
            active_layer = str(ready.get("activeLayer"))
            if active_layer != str(copy_layer["id"]):
                raise RuntimeError(
                    "Zmieniła się aktywna warstwa kopii. Przerwano przed zaznaczeniem."
                )
            # Odczyt czerwonego kanału maski (identycznego z G i B) z INNEGO dokumentu.
            # Jeśli ta wersja PhotoCraft nie obsługuje cross-document, kończymy
            # z otwartymi dokumentami do prostego manualnego Ctrl+klik RGB.
            try:
                text(
                    await session.call_tool(
                        "command_run",
                        arguments={
                            "id": "select.loadSelection",
                            "params": {
                                "channel": "red",
                                "document": mask_index,
                                "operation": "new",
                            },
                        },
                    )
                )
            except Exception as exc:
                print(
                    "SAM3: automatyczny odczyt kanału maski nie powiódł się: "
                    + str(exc),
                    flush=True,
                )
                print("SAM3: maska pozostaje otwarta jako osobny dokument.", flush=True)
                print(
                    "SAM3: w zakładce maski wybierz Kanały > Ctrl+klik RGB; wróć do kopii zdjęcia.",
                    flush=True,
                )
                print(
                    "SAM3: następnie Warstwa > Maska warstwy > Pokaż zaznaczenie.",
                    flush=True,
                )
                print("Wyciecie: " + str(cutout), flush=True)
                print("Maska: " + str(mask), flush=True)
                return 0
            inspected = decode(await session.call_tool("doc_inspect", arguments={}))
            bounds = inspected.get("selectionBounds")
            if (
                inspected.get("name") != ready.get("name")
                or not isinstance(bounds, (list, tuple))
                or len(bounds) != 4
            ):
                print(
                    "SAM3: wysłano select.loadSelection, lecz nie potwierdzono granic zaznaczenia.",
                    flush=True,
                )
                print(
                    "SAM3: sprawdź mrówki na kopii pełnego zdjęcia. W razie potrzeby wczytaj kanał ręcznie.",
                    flush=True,
                )
                print("Wyciecie: " + str(cutout), flush=True)
                print("Maska: " + str(mask), flush=True)
                return 0
            print("Zaznaczenie: " + str(bounds), flush=True)
            print(
                "   Próba automatycznej maski na kopii pełnego zdjęcia...", flush=True
            )
            try:
                ok, method = await attempt_reveal_selection(
                    session,
                    ready["name"],
                    copy_layer["id"],
                    [str(x["id"]) for x in stable_layers(ready)],
                )
            except Exception as exc:
                ok, method = False, "Bezpiecznie przerwano: " + str(exc)
            if ok:
                print("GOTOWE: SAM3 MASKA WARSTWY", flush=True)
                print("Utworzono edytowalną maskę: " + method, flush=True)
            else:
                print("GOTOWE: SAM3 ZAZNACZENIE", flush=True)
                print("Automatyczna maska niedostępna: " + method, flush=True)
                print(
                    "Ostatni krok ręczny: Warstwa > Maska warstwy > Pokaż zaznaczenie.",
                    flush=True,
                )
            print(
                "Warstwa: "
                + str(copy_layer.get("name"))
                + " ID "
                + str(copy_layer["id"]),
                flush=True,
            )
            print("Wyciecie: " + str(cutout), flush=True)
            print("Maska: " + str(mask), flush=True)
            print("Nie zapisano ani nie zmieniono źródłowego dokumentu.", flush=True)
            return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Most SAM3 → zaznaczenie PhotoCraft")
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    try:
        sys.exit(asyncio.run(main(args.prompt)))
    except Exception as exc:
        print("BŁĄD SAM3:", str(exc), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
