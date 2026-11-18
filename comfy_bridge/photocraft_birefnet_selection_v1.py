"""PhotoCraft zaznaczenie -> BiRefNet General -> nowa warstwa (wersja testowa).

Wymaga:
- PhotoCraft GUI z MCP pod http://127.0.0.1:8766/mcp
- ComfyUI pod http://127.0.0.1:8188
- dzialajacego comfy_birefnet_bridge.py i JSON obok niego
- pip install mcp (w srodowisku, z ktorego uruchamiasz ten skrypt)

Uruchom z katalogu glownego projektu: venv\\Scripts\\python.exe comfy_bridge\\photocraft_birefnet_selection_v1.py

UWAGA: eksportuje widoczny obraz, ALE do BiRefNet wysyla tylko zaznaczony prostokat.
Zaznaczenie nieprostokatne sprowadza do prostokata obejmujacego. Nie zmienia oryginalu.
Import w trybie bridge: doc_open wynikowego PNG -> select.all -> edit.copy ->
ui_menu_invoke window.document.<index> -> edit.paste. Przed testem zrob kopie projektu.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
import argparse
from pathlib import Path

MCP_URL = "http://127.0.0.1:8766/mcp"
BRIDGE_DIR = Path(__file__).resolve().parent
BRIDGE_SCRIPT = BRIDGE_DIR / "comfy_birefnet_bridge.py"
BASE_DIR = BRIDGE_DIR.parent

EXPORT_DIR = Path(os.environ.get("PHOTOCRAFT_EXPORT_DIR", str(BASE_DIR / "exports")))
IMPORT_DIR = Path(os.environ.get("PHOTOCRAFT_IMPORT_DIR", str(BASE_DIR / "import")))

COMFY_PYTHON = Path(os.environ.get("PHOTOCRAFT_COMFY_PYTHON", ""))


def result_text(result):
    text = "\n".join(part.text for part in result.content if part.type == "text")
    if result.isError:
        raise RuntimeError("MCP: " + text)
    return text


def inspect_json(result):
    return json.loads(result_text(result))


def pil_mode(command, full_path, output_path, box_values, cutout_path=None):
    """Wywoływane WYŁĄCZNIE przez Python ComfyUI z Pillow; bez MCP."""
    from PIL import Image

    x, y, w, h = [int(v) for v in box_values]
    with Image.open(full_path) as source:
        source.load()
        W, H = source.size
        if not (0 <= x < W and 0 <= y < H and 0 < w <= W - x and 0 < h <= H - y):
            raise ValueError("Zaznaczenie jest poza wymiarami eksportu PNG")
        if command == "crop":
            source.crop((x, y, x + w, y + h)).convert("RGB").save(output_path, "PNG")
        elif command == "compose":
            with Image.open(cutout_path) as extracted:
                extracted.load()
                if extracted.size != (w, h):
                    raise ValueError(
                        f"BiRefNet zwrocil {extracted.size}, oczekiwano {(w,h)}. Nie nakladam wyniku w ciemno."
                    )
                canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                canvas.paste(extracted.convert("RGBA"), (x, y))
                canvas.save(output_path, "PNG")
        else:
            raise ValueError("Nieznana akcja PIL")


def run_pil(command, full_path, output_path, box_values, cutout_path=None):
    if not COMFY_PYTHON.is_file():
        raise FileNotFoundError(
            "Nie znaleziono interpretera Python ComfyUI. "
            "Ustaw sciezke COMFY_ROOT w local_config.bat "
            "lub zmienna PHOTOCRAFT_COMFY_PYTHON. "
            f"Aktualna wartosc: {COMFY_PYTHON}"
        )
    args = [
        str(COMFY_PYTHON),
        str(Path(__file__).resolve()),
        "--pil",
        command,
        str(full_path),
        str(output_path),
        *[str(x) for x in box_values],
    ]
    if cutout_path is not None:
        args.append(str(cutout_path))
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Przygotowanie obrazu PIL: " + (proc.stdout + "\n" + proc.stderr)[-1500:]
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Nie powstal plik: " + str(output_path))


async def main():
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    if not BRIDGE_SCRIPT.is_file():
        raise FileNotFoundError(f"Brakuje dzialajacego mostu: {BRIDGE_SCRIPT}")
    if not EXPORT_DIR.is_dir():
        raise FileNotFoundError(f"Brak folderu eksportu PhotoCraft: {EXPORT_DIR}")
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    filename = f"pc_birefnet_input_{token}.png"
    input_file = EXPORT_DIR / filename
    print("1/4 Sprawdzam zaznaczenie i eksportuje widoczny dokument...", flush=True)

    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            print("   MCP: inicjalizacja sesji...", flush=True)
            await session.initialize()
            print("   MCP: sesja gotowa.", flush=True)
            print("   MCP: odczyt dokumentu...", flush=True)
            before = inspect_json(await session.call_tool("doc_inspect", arguments={}))
            print("   MCP: odczyt OK.", flush=True)
            layers = before.get("layers")
            if not before.get("name") or not isinstance(layers, list):
                raise RuntimeError("Nie ma aktywnego dokumentu PhotoCraft.")
            # Potwierdzone przez test: selectionBounds = [x, y, width, height].
            if before.get("hasSelection") is not True:
                raise RuntimeError(
                    "Brak aktywnego zaznaczenia. Zaznacz prostokat w PhotoCraft."
                )
            bounds = before.get("selectionBounds")
            W, H = before.get("width"), before.get("height")
            if (
                not isinstance(bounds, list)
                or len(bounds) != 4
                or any(type(v) is not int for v in bounds)
                or type(W) is not int
                or type(H) is not int
            ):
                raise RuntimeError(
                    "Nie mozna odczytac granic zaznaczenia z doc_inspect."
                )
            x, y, w, h = bounds
            if not (
                0 <= x < W and 0 <= y < H and 20 <= w <= W - x and 20 <= h <= H - y
            ):
                raise RuntimeError(
                    f"Nieprawidlowe granice zaznaczenia: {bounds}; obraz {W}x{H}"
                )
            # Uwaga: selectionBounds zawiera prostokat obejmujacy zaznaczenie.
            # Dla wyborow innych niz prostokat nie zachowujemy dokladnego ksztaltu zaznaczenia.
            print(f"   Zaznaczenie: x={x}, y={y}, szer={w}, wys={h}", flush=True)
            previous_ids = {str(x.get("id")) for x in layers if isinstance(x, dict)}
            original_name = before["name"]
            original_dimensions = (before.get("width"), before.get("height"))
            info = inspect_json(await session.call_tool("session_list", arguments={}))
            print(
                "   MCP: sesja dokumentow:",
                json.dumps(info, ensure_ascii=False)[:900],
                flush=True,
            )
            if not previous_ids:
                raise RuntimeError("Nie mozna odczytac warstw dokumentu.")
            print(
                "   MCP: eksport obrazu w trybie bridge (bez parametru format)...",
                flush=True,
            )
            # W trybie bridge PhotoCraft używa ustawień eksportu aplikacji.
            # Parametr format jest obsługiwany wyłącznie w trybie headless.
            export_msg = result_text(
                await session.call_tool("doc_export", arguments={"path": filename})
            )
            print("   MCP: odpowiedz na eksport: " + repr(export_msg[:300]), flush=True)
            if not input_file.is_file() or input_file.stat().st_size == 0:
                raise RuntimeError(
                    "PhotoCraft nie zapisal oczekiwanego pliku PNG: "
                    + str(input_file)
                    + "\nW trybie bridge sprawdz ustawienia eksportu i dozwolony folder zapisu W APLIKACJI PhotoCraft. "
                    + "Mozliwe tez, ze PhotoCraft wybral inny format lub lokalizacje. MCP: "
                    + export_msg[:300]
                )
            # Nie wysyłaj do BiRefNet pliku z błędnym rozszerzeniem/formatem.
            with input_file.open("rb") as f:
                header = f.read(8)
            if header != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError(
                    "Plik z eksportu nie jest PNG mimo rozszerzenia .png. "
                    + "Ustaw eksport PNG w PhotoCraft i ponow test."
                )
            cropped_input = EXPORT_DIR / f"pc_selection_crop_{token}.png"
            print("   Przycinam eksport do zaznaczonego prostokata...", flush=True)
            await asyncio.to_thread(run_pil, "crop", input_file, cropped_input, bounds)
            print("2/4 BiRefNet w ComfyUI — czekam na wynik...", flush=True)
            completed = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, str(BRIDGE_SCRIPT), str(cropped_input)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Blad mostu BiRefNet:\n"
                    + completed.stdout[-1200:]
                    + completed.stderr[-1200:]
                )
            # Most tworzy pojedynczy katalog birefnet_<id> i wypisuje sciezki.
            cutouts = [
                Path(line.strip())
                for line in completed.stdout.splitlines()
                if line.strip().lower().endswith("cutout.png")
            ]
            masks = [
                Path(line.strip())
                for line in completed.stdout.splitlines()
                if line.strip().lower().endswith("mask.png")
            ]
            if (
                len(cutouts) != 1
                or len(masks) != 1
                or not cutouts[0].is_file()
                or not masks[0].is_file()
            ):
                raise RuntimeError(
                    "Nie rozpoznano pary cutout.png + mask.png. Wynik mostu:\n"
                    + completed.stdout[-1500:]
                )
            print("3/4 Przygotowuje wyciety PNG do importu...")
            import_file = IMPORT_DIR / f"pc_birefnet_cutout_{token}.png"
            await asyncio.to_thread(
                run_pil, "compose", input_file, import_file, bounds, cutouts[0]
            )
            print(
                "   Przygotowano przezroczysty PNG w rozmiarze calego dokumentu.",
                flush=True,
            )
            # Nie kontynuuj, gdy w trakcie pracy uzytkownik zmienil dokument.
            check = inspect_json(await session.call_tool("doc_inspect", arguments={}))
            if (
                check.get("name") != before.get("name")
                or check.get("width") != before.get("width")
                or check.get("height") != before.get("height")
                or check.get("selectionBounds") != bounds
            ):
                raise RuntimeError(
                    "Zmienil sie aktywny dokument PhotoCraft; wynik pozostaje na dysku: "
                    + str(import_file)
                )
            print(
                "4/4 Import przez otwarcie PNG, kopiowanie i wklejenie do oryginalu..."
            )
            # Zbadamy schematy przed zmianami. Bez nich nie zakladamy, ze dany serwer
            # udostepnia doc_open/doc_select/doc_close i odpowiednie komendy.
            exposed = {t.name: t for t in (await session.list_tools()).tools}
            for needed in (
                "doc_open",
                "ui_menu_invoke",
                "command_run",
                "doc_inspect",
                "session_list",
            ):
                if needed not in exposed:
                    raise RuntimeError(
                        "MCP nie udostepnia "
                        + needed
                        + "; wyniki pozostaja na dysku: "
                        + str(import_file)
                    )

            # Index oryginalnego dokumentu ustalamy z session_list; zakladanie 0
            # groziloby wklejeniem do innego otwartego obrazu.
            def get_documents(info):
                candidate = info.get("documents")
                if isinstance(candidate, list):
                    return candidate
                session_data = info.get("session")
                if isinstance(session_data, dict) and isinstance(
                    session_data.get("documents"), list
                ):
                    return session_data["documents"]
                return None

            docs = get_documents(info)
            original_index = info.get("active")
            if original_index is None and isinstance(info.get("session"), dict):
                original_index = info["session"].get("active")
            if (
                not isinstance(docs, list)
                or type(original_index) is not int
                or not 0 <= original_index < len(docs)
            ):
                raise RuntimeError(
                    "Nie potrafie jednoznacznie ustalic indeksu oryginalnego dokumentu. Wyniki: "
                    + str(import_file)
                )
            print(
                f"   Oryginalny dokument: indeks {original_index}; pozostaje niezapisany.",
                flush=True,
            )
            # doc_open wymaga --automation-read-root w GUI (launcher).
            try:
                open_response = result_text(
                    await session.call_tool(
                        "doc_open", arguments={"path": str(import_file)}
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    "Nie udalo sie otworzyc wynikowego PNG w PhotoCraft. "
                    + f"Sprawdz --automation-read-root dla katalogu {IMPORT_DIR} w launcherze. "
                    + str(exc)
                ) from exc
            print("   Otwarto wynikowe PNG:", open_response[:200], flush=True)
            new_info = inspect_json(
                await session.call_tool("session_list", arguments={})
            )
            new_docs = get_documents(new_info)
            new_index = new_info.get("active")
            if new_index is None and isinstance(new_info.get("session"), dict):
                new_index = new_info["session"].get("active")
            if (
                not isinstance(new_docs, list)
                or len(new_docs) != len(docs) + 1
                or type(new_index) is not int
                or not 0 <= new_index < len(new_docs)
                or new_index == original_index
            ):
                raise RuntimeError(
                    "PNG otwarto, ale nie da sie potwierdzic aktywnego drugiego dokumentu. Nic nie kopiowano."
                )
            png_doc = inspect_json(await session.call_tool("doc_inspect", arguments={}))
            if (png_doc.get("width"), png_doc.get("height")) != original_dimensions:
                raise RuntimeError(
                    "PNG ma inne wymiary niz oryginal. Przerwano przed wklejeniem. "
                    + f'PNG: {png_doc.get("width")}x{png_doc.get("height")}, '
                    + f"oryginal: {original_dimensions}."
                )
            # Komendy bez ścieżek plikowych są dozwolone w trybie bridge.
            result_text(
                await session.call_tool(
                    "command_run", arguments={"id": "select.all", "params": {}}
                )
            )
            result_text(
                await session.call_tool(
                    "command_run", arguments={"id": "edit.copy", "params": {}}
                )
            )
            # Przełączenie przez Window > dokument (bridge GUI); bez zamykania i zapisywania.
            result_text(
                await session.call_tool(
                    "ui_menu_invoke",
                    arguments={"id": f"window.document.{original_index}"},
                )
            )
            selected = inspect_json(
                await session.call_tool("doc_inspect", arguments={})
            )
            if (
                selected.get("name") != original_name
                or (selected.get("width"), selected.get("height"))
                != original_dimensions
                or {str(x.get("id")) for x in selected.get("layers", [])}
                != previous_ids
            ):
                raise RuntimeError(
                    "Po przełączeniu nie potwierdzono oryginalnego dokumentu. Nie wklejano obrazu."
                )
            print("   Wklejam skopiowane PNG jako nowa warstwe...", flush=True)
            # Wklejanie na srodku płótna; przy identycznych rozmiarach eksportu
            # powinno zachować pozycję, ale wynik trzeba ocenić wizualnie.
            result_text(
                await session.call_tool(
                    "command_run",
                    arguments={
                        "id": "edit.paste",
                        "params": {
                            "center": [
                                original_dimensions[0] // 2,
                                original_dimensions[1] // 2,
                            ]
                        },
                    },
                )
            )
            after = inspect_json(await session.call_tool("doc_inspect", arguments={}))
            added = [
                x
                for x in after.get("layers", [])
                if isinstance(x, dict) and str(x.get("id")) not in previous_ids
            ]
            if after.get("name") != original_name or len(added) != 1:
                print(
                    "UWAGA: wykonano wklejenie, ale nie potwierdzono dokladnie jednej warstwy."
                )
                print("Wyciecie:", cutouts[0], "Maska:", masks[0], sep="\n")
                return 2
            print(
                f"GOTOWE: nowa warstwa '{added[0].get('name')}' (ID {added[0].get('id')})."
            )
            print("Wyciecie:", cutouts[0])
            print("Maska:", masks[0])
            print(
                "Nie zapisano dokumentu PhotoCraft. Wynikowe PNG pozostalo tez otwarte jako osobny dokument."
            )
            print("Sprawdz czy przezroczystosc i polozenie sa prawidlowe.")
            return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--pil":
        try:
            if len(sys.argv) not in (9, 10):
                raise ValueError("Niepoprawne argumenty wewnetrznego trybu PIL.")
            pil_mode(
                sys.argv[2],
                Path(sys.argv[3]),
                Path(sys.argv[4]),
                sys.argv[5:9],
                Path(sys.argv[9]) if len(sys.argv) == 10 else None,
            )
        except Exception as exc:
            print("BLAD PIL:", exc, file=sys.stderr)
            raise SystemExit(1)
        raise SystemExit(0)
    try:
        status = asyncio.run(main())
    except Exception as exc:
        import traceback

        print("BLAD:", exc, file=sys.stderr)
        print(
            "--- PELNA DIAGNOSTYKA (w tym wewnetrzne bledy TaskGroup) ---",
            file=sys.stderr,
        )
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        raise SystemExit(1)
    else:
        raise SystemExit(status)
