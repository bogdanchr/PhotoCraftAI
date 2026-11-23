# -*- coding: utf-8 -*-
"""Read-only diagnostic of PhotoCraft MCP state: ui_inspect and doc_inspect.
No command_run, control_call, ui_set or document edits.
"""
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'diagnostyka_mcp'
URL = 'http://127.0.0.1:8766/mcp'


def payload(result):
    if getattr(result, 'isError', getattr(result, 'is_error', False)):
        raise RuntimeError('MCP error: ' + ' '.join(getattr(p, 'text', '') for p in result.content))
    obj = {}
    if getattr(result, 'structuredContent', None) is not None:
        obj['structuredContent'] = result.structuredContent
    for p in result.content:
        if getattr(p, 'type', '') != 'text':
            continue
        raw = p.text
        try:
            val = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            val = raw
        obj.setdefault('content', []).append(val)
    return obj


async def read_state():
    async with streamable_http_client(URL) as channels:
        read, write = channels[:2]  # MCP variants return 2 or 3 context values
        async with ClientSession(read, write) as session:
            await session.initialize()
            answers = {}
            for name in ('ui_inspect', 'doc_inspect'):
                print('Odczyt:', name, flush=True)
                try:
                    answers[name] = payload(await session.call_tool(name, arguments={}))
                except Exception as exc:
                    answers[name] = {'error': str(exc)}
            return answers


def main():
    print('PhotoCraft MCP: diagnostyka tylko do odczytu')
    print('Adres:', URL)
    try:
        data = asyncio.run(read_state())
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / ('stan_ui_dokumentu_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        print('\nZapisano:', path)
        for name, result in data.items():
            print(name + ':', 'BLAD: ' + result['error'] if 'error' in result else 'OK')
        print('Wyslij plik JSON do analizy. Nie wykonano zmian.')
        return 0
    except Exception as exc:
        print('BLAD:', exc, file=sys.stderr)
        print('Sprawdz, czy PhotoCraft i MCP sa uruchomione.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
