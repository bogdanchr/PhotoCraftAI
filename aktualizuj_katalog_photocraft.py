# -*- coding: utf-8 -*-
"""Pobiera katalog command_list z lokalnego PhotoCraft MCP i porownuje wersje.
Nie wysyla command_run ani nie modyfikuje dokumentow PhotoCraft.
"""
import asyncio
import ast
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

BASE_DIR = Path(__file__).resolve().parent
CATALOG = BASE_DIR / 'polecenia_photocraft.txt'
ARCHIVE = BASE_DIR / 'historia_katalogow'
MCP_URL = 'http://127.0.0.1:8766/mcp'


def parse_catalog_text(raw):
    """Stare zrzuty TextContent(...) i zwykle JSON, zgodnie z app.py."""
    raw = raw.lstrip('\ufeff').strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        marker = "TextContent(type='text', text="
        pos = raw.find(marker)
        if pos < 0:
            raise ValueError('Nie rozpoznano poprzedniego katalogu (to nie jest JSON).')
        # Stare zrzuty: TextContent(type='text', text='[....]', ...)
        import io
        import tokenize
        start = pos + len(marker)
        token = next((t for t in tokenize.generate_tokens(io.StringIO(raw[start:]).readline)
                      if t.type == tokenize.STRING), None)
        if token is None:
            raise ValueError('Nie udalo sie odczytac zapisu TextContent.')
        obj = json.loads(ast.literal_eval(token.string))
    if isinstance(obj, dict):
        obj = obj.get('commands', obj.get('result', obj))
    if not isinstance(obj, list):
        raise ValueError('Katalog nie jest lista komend.')
    if not obj or any(not isinstance(c, dict) or not isinstance(c.get('id'), str)
                      or not c['id'].strip() for c in obj):
        raise ValueError('Katalog jest pusty lub ma niepoprawna strukture.')
    ids = [c['id'] for c in obj]
    if len(ids) != len(set(ids)):
        raise ValueError('W katalogu sa powtorzone ID; przerwano, by nie nadpisac starej wersji.')
    return obj


def load_previous():
    if not CATALOG.exists():
        return None
    return parse_catalog_text(CATALOG.read_text(encoding='utf-8-sig'))


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def explain_difference(old, new):
    if old is None:
        return 'Nie znaleziono poprzedniego katalogu. To pierwsze pobranie.\n'
    old_map = {c['id']: c for c in old}
    new_map = {c['id']: c for c in new}
    added = sorted(new_map.keys() - old_map.keys())
    removed = sorted(old_map.keys() - new_map.keys())
    changed = sorted(i for i in new_map.keys() & old_map.keys()
                     if stable_json(new_map[i]) != stable_json(old_map[i]))
    lines = [f'Poprzedni katalog: {len(old)} komend',
             f'Nowy katalog: {len(new)} komend',
             f'Dodane: {len(added)} | Usuniete: {len(removed)} | Zmienione: {len(changed)}', '']
    lines.extend(['=== DODANE ==='] + (added or ['(brak)']))
    lines.extend(['', '=== USUNIETE ==='] + (removed or ['(brak)']))
    lines.append('')
    lines.append('=== ZMIENIONE (nazwy pol i wartosci) ===')
    if not changed:
        lines.append('(brak)')
    for ident in changed:
        before, after = old_map[ident], new_map[ident]
        changed_fields = sorted(k for k in set(before) | set(after)
                                if stable_json(before.get(k)) != stable_json(after.get(k)))
        lines.append(f'- {ident}: {", ".join(changed_fields)}')
        for field in changed_fields:
            lines.append(f'    {field}: STARE={stable_json(before.get(field))[:600]}')
            lines.append(f'           NOWE={stable_json(after.get(field))[:600]}')
    return '\n'.join(lines) + '\n'


async def get_new_catalog():
    async with streamable_http_client(MCP_URL) as streams:
        # Rozne wersje mcp zwracaja 2 lub 3 elementy.
        read, write = streams[:2]
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool('command_list', arguments={})
            if getattr(result, "isError", getattr(result, "is_error", False)):
                raise RuntimeError('command_list zwrocilo blad: ' + ' '.join(
                    x.text for x in result.content if x.type == 'text')[:1000])
            texts = [part.text for part in result.content if part.type == 'text']
            if not texts:
                # Nieprobujemy zgadywac alternatywnych struktur, aby nie nadpisac katalogu.
                raise ValueError('command_list nie zwrocilo tekstowego JSON.')
            # Wiekszosc serwerow MCP zwraca cala liste w jednym TextContent.
            # Dopuszczamy kilka blokow stanowiacuch osobne listy JSON.
            catalog = []
            for text in texts:
                catalog.extend(parse_catalog_text(text))
            parse_catalog_text(json.dumps(catalog, ensure_ascii=False))
            return catalog


def main():
    print('PhotoCraft: odczyt polecen przez MCP (bez edycji dokumentu).', flush=True)
    print('Adres MCP:', MCP_URL, flush=True)
    try:
        new_catalog = asyncio.run(get_new_catalog())
        old_catalog = load_previous()  # przed zapisaniem, brak utraty nieznanego formatu
        # Chron przed przypadkowa zamiana pelnego katalogu na niepelny odczyt.
        if old_catalog and len(new_catalog) < len(old_catalog) * 0.90:
            raise ValueError(
                f"Nowy katalog ma tylko {len(new_catalog)} komend, "
                f"poprzedni {len(old_catalog)}. Odmowa nadpisania (spadek >10%)."
            )
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if CATALOG.exists():
            previous_path = ARCHIVE / f'polecenia_PRZED_{timestamp}.txt'
            shutil.copy2(CATALOG, previous_path)
            print('Zachowano poprzedni katalog:', previous_path)
        report = explain_difference(old_catalog, new_catalog)
        report_path = ARCHIVE / f'porownanie_{timestamp}.txt'
        report_path.write_text(report, encoding='utf-8-sig')
        archive_new = ARCHIVE / f'polecenia_{timestamp}.json'
        new_text = json.dumps(new_catalog, ensure_ascii=False, indent=2)
        archive_new.write_text(new_text, encoding='utf-8')
        # Wstawiamy nowa liste do istniejacej sciezki rozpoznawanej przez Gradio.
        temp_file = CATALOG.with_suffix('.tmp')
        temp_file.write_text(new_text, encoding='utf-8')
        temp_file.replace(CATALOG)
        print('\n' + report.split('=== DODANE ===')[0])
        print('Nowy katalog:', CATALOG)
        print('Raport porownania:', report_path)
        print('Archiwum nowego katalogu:', archive_new)
        print('GOTOWE. Mozesz teraz uruchomic Gradio.')
        return 0
    except Exception as exc:
        print('\nBLAD:', str(exc), file=sys.stderr)
        print('Nie nadpisano starego katalogu (chyba ze doszlo do bledu zapisu dysku).', file=sys.stderr)
        print('Sprawdz, czy PhotoCraft i jego serwer MCP sa uruchomione.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
