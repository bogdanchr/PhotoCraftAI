# -*- coding: utf-8 -*-
"""Bezpieczny audyt mozliwosci kadrowania i prostowania w PhotoCraft MCP.

Odczyt command_list, bez command_run, bez edycji i zapisu dokumentow.
W razie braku MCP mozna uzyc --offline dla istniejacego katalogu.
"""
import argparse
import ast
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MCP_URL = 'http://127.0.0.1:8766/mcp'
CATEGORIES = {
    'Kadrowanie': [r'crop', r'kadrow', r'przytn', r'przycina'],
    'Prostowanie i obrót': [r'straighten', r'rotate', r'rotation', r'angle', r'obr[oó]t', r'obr[oó]c', r'prostow', r'pochyl'],
    'Geometria i transformacje': [r'transform', r'perspective', r'distort', r'warp', r'perspektyw', r'deform', r'skalow'],
    'Rozmiar i płótno': [r'resize', r'image size', r'canvas size', r'resample', r'rozmiar', r'rozdzielcz', r'canvas'],
    'Narzędzia pomocnicze': [r'grid', r'ruler', r'guide', r'align', r'wyrown', r'wyr[oó]wn', r'select\.', r'selection'],
}


def parse_catalog(raw):
    raw = raw.lstrip('\ufeff').strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        marker = "TextContent(type='text', text="
        if marker not in raw:
            raise ValueError('Nieznany format katalogu: oczekiwano JSON lub TextContent')
        import tokenize
        import io
        start = raw.index(marker) + len(marker)
        token = next((t for t in tokenize.generate_tokens(io.StringIO(raw[start:]).readline) if t.type == tokenize.STRING), None)
        if token is None:
            raise ValueError('Brak tekstu w TextContent')
        value = json.loads(ast.literal_eval(token.string))
    if isinstance(value, dict):
        value = value.get('commands', value.get('result', value))
    if not isinstance(value, list) or not value:
        raise ValueError('Katalog pusty lub nie jest listą')
    if any(not isinstance(x, dict) or not isinstance(x.get('id'), str) for x in value):
        raise ValueError('Nieprawidłowe rekordy poleceń')
    return value


async def fetch():
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async with streamable_http_client(MCP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            result = await session.call_tool('command_list', arguments={})
            if getattr(result, 'isError', False):
                raise RuntimeError('MCP command_list zwróciło błąd')
            parts = [x.text for x in result.content if getattr(x, 'type', None) == 'text']
            if not parts:
                raise ValueError('command_list nie zwróciło tekstu')
            commands = []
            for part in parts:
                commands.extend(parse_catalog(part))
            return commands


def classify(command):
    # ID, label, menu i opis mogą zawierać nazwy funkcji; argumenty traktujemy tylko jako sygnał dodatkowy.
    fields = [command.get('id', ''), command.get('label', ''), command.get('description', ''), command.get('menu', '')]
    haystack = ' '.join(str(x) for x in fields).casefold()
    matches = [category for category, patterns in CATEGORIES.items() if any(re.search(p, haystack, re.I) for p in patterns)]
    if not matches:
        # Zawartość schematu pod uwagę jedynie dla potencjalnych komend transformacji/obrazu.
        ident = command.get('id', '').casefold()
        if ident.startswith(('image.', 'edit.', 'layer.', 'view.', 'select.')):
            params = str(command.get('params', '')).casefold()
            if any(re.search(p, params, re.I) for p in (r'crop', r'straighten', r'rotation', r'resample', r'perspective')):
                matches = ['Inne — kandydat z parametrów']
    return matches


def report(catalog, source):
    groups = {key: [] for key in [*CATEGORIES, 'Inne — kandydat z parametrów']}
    for cmd in catalog:
        for category in classify(cmd):
            groups[category].append(cmd)
    lines = [
        'AUDYT PHOTOCRAFT MCP — KADROWANIE I PROSTOWANIE',
        f'Źródło: {source}',
        f'Liczba komend w katalogu: {len(catalog)}',
        'TYLKO ODCZYT: nie uruchomiono żadnej komendy edycyjnej.',
        'Ważne: enabled może zależeć od aktywnego dokumentu i zaznaczenia.',
        'Sam schemat komendy NIE potwierdza niedestrukcyjności ani jakości operacji.',
        '',
    ]
    for group, commands in groups.items():
        lines += [f'=== {group.upper()} ({len(commands)}) ===']
        for cmd in sorted(commands, key=lambda c: c['id']):
            lines.append(f"ID: {cmd['id']}")
            lines.append(f"  Etykieta: {cmd.get('label', '')}")
            lines.append(f"  Enabled w chwili odczytu: {cmd.get('enabled', 'brak danych')}")
            if cmd.get('menu'):
                lines.append('  Menu: ' + json.dumps(cmd['menu'], ensure_ascii=False))
            if cmd.get('description'):
                lines.append('  Opis: ' + str(cmd['description']))
            lines.append('  Parametry: ' + json.dumps(cmd.get('params', 'brak danych'), ensure_ascii=False, indent=2, default=str))
            lines.append('')
        lines.append('')
    lines += [
        'PYTANIA DO NASTĘPNEGO ETAPU (wymagają testu na kopii):',
        '1. Czy crop nie usuwa pikseli i można przywrócić cały obraz?',
        '2. Czy straighten zmienia tylko geometrię kadru, czy również przelicza piksele?',
        '3. Czy można ustawić precyzyjny kąt / proporcję / prostokąt kadru?',
        '4. Czy operacja ma podgląd i można ją anulować przed zastosowaniem?',
        '5. Czy maski i istniejące warstwy zachowują poprawne położenie po operacji?',
    ]
    return '\n'.join(lines), groups


def main():
    parser = argparse.ArgumentParser(description='Tylko odczyt command_list PhotoCraft MCP')
    parser.add_argument('--offline', action='store_true', help='Czytaj istniejący polecenia_photocraft.txt zamiast łączyć MCP')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'polecenia_photocraft.txt')
    args = parser.parse_args()
    try:
        if args.offline:
            catalog = parse_catalog(args.catalog.read_text(encoding='utf-8-sig'))
            source = str(args.catalog)
        else:
            catalog = asyncio.run(fetch())
            source = f'żywy MCP: {MCP_URL}'
        text, groups = report(catalog, source)
        dest = ROOT / 'audyt_kadrowania_MCP.txt'
        dest.write_text(text, encoding='utf-8-sig')
        # Zapis tylko raportu, nie zmienia katalogu i nie dotyka zdjęć.
        print(text[:5000])
        print('\nZAPISANO RAPORT:', dest)
        print('Dopasowane polecenia:', sum(len(v) for v in groups.values()))
        return 0
    except Exception as exc:
        print('BŁĄD ODCZYTU:', str(exc), file=sys.stderr)
        print('Nie wykonano żadnej operacji edycyjnej.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
