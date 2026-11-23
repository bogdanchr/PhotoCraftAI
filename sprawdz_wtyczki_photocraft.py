"""Bezpieczny odczyt obsługi wtyczek PhotoCraft przez lokalny MCP.
Nie instaluje, nie uruchamia filtrów, nie zapisuje dokumentu.
Uruchom: D:\\PhotoCraftAI\\venv\\Scripts\\python.exe sprawdz_wtyczki_photocraft.py
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = 'http://127.0.0.1:8766/mcp'
PLUGIN_IDS = ('plugin.list','plugin.install','plugin.run','plugin.remove','plugin.reload')


def extract(result):
    payload = '\n'.join(getattr(x,'text','') for x in result.content if getattr(x,'type',None)=='text')
    if getattr(result,'isError',False):
        raise RuntimeError(payload)
    try:
        return json.loads(payload)
    except (ValueError, TypeError):
        return payload


def unwrap_commands(data):
    if isinstance(data,list):
        return data
    if isinstance(data,dict):
        for key in ('commands','result','data'):
            if key in data:
                value = unwrap_commands(data[key])
                if isinstance(value,list):
                    return value
    return []


async def main():
    report = {'generated_at':datetime.now().astimezone().isoformat(),'mcp':MCP_URL,
              'note':'Only command_list and plugin.list (read-only) were called.'}
    async with streamable_http_client(MCP_URL) as streams:
        read,write = streams[:2]
        async with ClientSession(read,write) as session:
            await session.initialize()
            print('Połączono z PhotoCraft MCP.')
            try:
                commands = unwrap_commands(extract(await session.call_tool('command_list',arguments={})))
                matching = [x for x in commands if isinstance(x,dict) and str(x.get('id','')).startswith('plugin.')]
                report['catalog_total'] = len(commands)
                report['plugin_commands'] = matching
                print('Komendy w katalogu:', len(commands))
                print('Komendy plugin.*:', ', '.join(x['id'] for x in matching) or '(brak)')
            except Exception as exc:
                report['catalog_error'] = str(exc)
                print('Nie udało się odczytać command_list:',exc)
                matching=[]
            if any(x.get('id') == 'plugin.list' for x in matching):
                try:
                    plugins = extract(await session.call_tool('command_run',arguments={'id':'plugin.list','params':{}}))
                    report['plugin_list_result'] = plugins
                    print('\nWynik plugin.list:')
                    print(json.dumps(plugins,ensure_ascii=False,indent=2)[:10000] if not isinstance(plugins,str) else plugins[:10000])
                except Exception as exc:
                    report['plugin_list_error'] = str(exc)
                    print('plugin.list zgłosiło błąd:',exc)
            else:
                print('Nie uruchamiam plugin.list: nie potwierdzono komendy w katalogu.')
    destination = Path(__file__).with_name('raport_wtyczki_photocraft.json')
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nRaport:',destination)
    print('Nie zmieniono dokumentu ani wtyczek.')


if __name__=='__main__':
    try:
        asyncio.run(main())
    except Exception as exc:
        print('Błąd połączenia:',exc)
        raise SystemExit(1)
