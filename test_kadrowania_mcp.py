# -*- coding: utf-8 -*-
"""Jednorazowy, kontrolowany test image.crop na kopii zdjęcia.
Domyślnie tylko odczyt + plan. --apply wymaga ręcznej zgody.
Nie zapisuje pliku, nie wykonuje Undo i nie powtarza command_run.
"""
import argparse
import asyncio
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8766/mcp'
OUT = ROOT / 'raport_testu_kadrowania_MCP.txt'


def parse_result(result):
    if getattr(result, 'isError', getattr(result, 'is_error', False)):
        raise RuntimeError('MCP zwróciło błąd: ' + ' '.join(getattr(x, 'text', '') for x in result.content)[:1000])
    structured = getattr(result, 'structuredContent', None)
    if isinstance(structured, dict) and 'name' in structured and 'width' in structured:
        return structured
    text_blocks = [p.text for p in result.content if getattr(p, 'type', None) == 'text']
    for text in text_blocks:
        try:
            data = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data
    raise ValueError('Nie udało się odczytać JSON z MCP')


def summary(data):
    # Na potrzeby kontroli bezpieczeństwa nie zgadujemy struktury, gdy brak pól.
    name, width, height = data.get('name'), data.get('width'), data.get('height')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Brak nazwy aktywnego dokumentu')
    if type(width) is not int or type(height) is not int or width < 100 or height < 100:
        raise ValueError('Brak wiarygodnych wymiarów dokumentu')
    layers = data.get('layers')
    if not isinstance(layers, list) or len(layers) != 1 or not isinstance(layers[0], dict):
        raise ValueError('Dla bezpieczeństwa test wymaga dokładnie jednej warstwy (oryginalne testowe zdjęcie)')
    return {'name': name, 'width': width, 'height': height,
            'layers': [{'id': layers[0].get('id'), 'name': layers[0].get('name'), 'kind': layers[0].get('kind')}]}


def plan(width, height):
    # Kadr centralny 80% x 80% (bez zmiany proporcji), całkowite piksele.
    left, top = round(width * .10), round(height * .10)
    right, bottom = round(width * .90), round(height * .90)
    return {'x': left, 'y': top, 'width': right-left, 'height': bottom-top,
            'angle': 0, 'deleteCroppedPixels': False, 'contentAware': False}


def normalize_name(value):
    # Nazwa dokumentu może zawierać spacje, podkreślenia lub myślniki.
    value = unicodedata.normalize('NFKD', value.casefold())
    value = ''.join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r'[^a-z0-9]+', ' ', value).strip()


def is_our_test_image(name):
    name = normalize_name(name)
    return 'przyjacielski selfie' in name and 'parkowej' in name


def explain_exception(exc):
    # AnyIO/MCP wraps original exceptions as TaskGroup ExceptionGroup.
    if isinstance(exc, BaseExceptionGroup):
        parts = [explain_exception(inner) for inner in exc.exceptions]
        return ' | '.join(parts)[:1600]
    return f'{type(exc).__name__}: {exc}'


def emit(lines, line):
    print(line, flush=True)
    lines.append(line)


async def run(apply):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    lines = ['TEST PHOTOCRAFT MCP: image.crop', f'Data: {datetime.now().isoformat(timespec="seconds")}',
             f'Tryb: {"JEDNORAZOWA OPERACJA" if apply else "TYLKO ODCZYT"}']
    try:
        async with streamable_http_client(URL) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                before_raw = parse_result(await session.call_tool('doc_inspect', arguments={}))
                before = summary(before_raw)
                emit(lines, 'Dokument: ' + json.dumps(before, ensure_ascii=False))
                if not is_our_test_image(before['name']):
                    raise ValueError('Otwórz testowe zdjęcie przyjacielski selfie na parkowej ławce (spacje lub podkreślenia w nazwie są dozwolone). Innych dokumentów nie zmieniam.')
                params = plan(before['width'], before['height'])
                emit(lines, 'Plan image.crop: ' + json.dumps(params, ensure_ascii=False))
                emit(lines, 'Przewidywane wymiary płótna: ' + str((params['width'], params['height'])))
                if not apply:
                    emit(lines, 'TRYB ODCZYTU: brak operacji edycyjnej. Uruchom ponownie z --apply, jeśli chcesz testować.')
                    return lines
                emit(lines, 'UWAGA: operacja zmieni aktywny dokument w pamięci. Nie zapisuje go na dysk.')
                emit(lines, 'Przy przycięciu Background może zmienić się w Layer 0. To oczekiwane według katalogu.')
                consent = input('Wpisz dokładnie KADRUJ, aby wykonać JEDNĄ operację: ').strip()
                if consent != 'KADRUJ':
                    emit(lines, 'ANULOWANO: brak jednoznacznej zgody.')
                    return lines
                # Ponownie sprawdzamy dokument: bez cichego przełączenia zdjęcia.
                latest = summary(parse_result(await session.call_tool('doc_inspect', arguments={})))
                if latest != before:
                    raise ValueError('Stan dokumentu zmienił się od odczytu. Nie wykonano kadrowania.')
                emit(lines, 'WYSYŁAM command_run image.crop dokładnie jeden raz.')
                try:
                    cmd = await session.call_tool('command_run', arguments={'id': 'image.crop', 'params': params})
                    emit(lines, 'Wynik: ' + json.dumps(parse_result(cmd), ensure_ascii=False, default=str)[:1500])
                except Exception as exc:
                    emit(lines, 'UWAGA: wynik operacji niepewny: ' + str(exc))
                    emit(lines, 'NIE POWTARZAJ automatycznie. Sprawdź PhotoCraft ręcznie.')
                    return lines
                try:
                    after_raw = parse_result(await session.call_tool('doc_inspect', arguments={}))
                    after = summary(after_raw)
                    emit(lines, 'Po operacji: ' + json.dumps(after, ensure_ascii=False))
                    if (after['name'] == before['name'] and after['width'] == params['width']
                            and after['height'] == params['height']):
                        emit(lines, 'OK: wymiary nowego kadru potwierdzone. Zachowanie pikseli poza kadrem wymaga ręcznego testu!')
                    else:
                        emit(lines, 'UWAGA: nie potwierdzono oczekiwanego stanu. Sprawdź PhotoCraft ręcznie.')
                except Exception as exc:
                    emit(lines, 'UWAGA: brak odczytu stanu po operacji: ' + str(exc))
                emit(lines, 'RĘCZNA KONTROLA: rozszerz ponownie kadr w PhotoCraft i sprawdź, czy wracają piksele poza przycięciem.')
                emit(lines, 'Nie wykonano zapisu ani Undo. Nie uruchamiaj testu ponownie na już przyciętym dokumencie.')
                return lines
    except Exception as exc:
        emit(lines, 'PRZERWANO: ' + explain_exception(exc))
        return lines
    finally:
        OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')
        print('Raport:', OUT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true', help='Po wyraźnym zatwierdzeniu wykonaj jedno kadrowanie')
    args = p.parse_args()
    asyncio.run(run(args.apply))


if __name__ == '__main__':
    main()
