"""PhotoCraft AI — test odczytowy obsługi EDYCJI istniejącej warstwy Krzywe.
Nie zmienia dokumentu ani parametrów. Wymaga działającego PhotoCraft MCP.
Uruchom: python sprawdz_edycje_krzywych_v2_7_5.py
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = 'http://127.0.0.1:8766/mcp'
RELEVANT = ('layer.setAdjustment', 'layer.setProps', 'layer.newAdjustmentLayer.curves')


def unpack(result):
    raw = '\n'.join(x.text for x in result.content if getattr(x, 'type', '') == 'text')
    if getattr(result, 'isError', False):
        raise RuntimeError(raw)
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def command_items(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('commands', 'result', 'data'):
            if key in data:
                found = command_items(data[key])
                if isinstance(found, list):
                    return found
    return None


def find_layers(data):
    if isinstance(data, dict):
        if isinstance(data.get('layers'), list):
            return data['layers']
        for value in data.values():
            found = find_layers(value)
            if found is not None:
                return found
    if isinstance(data, list):
        for value in data:
            found = find_layers(value)
            if found is not None:
                return found
    return None


async def main():
    report = {
        'when': datetime.now().astimezone().isoformat(),
        'test_mode': 'read_only',
        'operations': ['command_list', 'doc_inspect'],
    }
    async with streamable_http_client(MCP_URL) as streams:
        r, w = streams[:2]
        async with ClientSession(r, w) as session:
            await session.initialize()
            commands = command_items(unpack(await session.call_tool('command_list', arguments={})))
            if commands is None:
                raise ValueError('command_list nie zwrócił katalogu komend')
            print('Liczba komend:', len(commands))
            report['commands'] = {}
            for name in RELEVANT:
                found = next((item for item in commands if isinstance(item, dict) and item.get('id') == name), None)
                report['commands'][name] = found
                print('\n' + name + ':')
                if found is None:
                    print('BRAK w katalogu')
                else:
                    print('enabled:', found.get('enabled'))
                    print('params:', found.get('params', '(brak)'))
                    if found.get('disabledReason'):
                        print('disabledReason:', found['disabledReason'])
            try:
                doc = unpack(await session.call_tool('doc_inspect', arguments={}))
                layers = find_layers(doc)
                report['document_state'] = {
                    'active_layer_id': doc.get('activeLayer') if isinstance(doc, dict) else None,
                    'layers': [],
                }
                for layer in layers or []:
                    if not isinstance(layer, dict):
                        continue
                    record = {
                        'id': layer.get('id'),
                        'name': layer.get('name'),
                        'kind': layer.get('kind'),
                        'selected': layer.get('selected'),
                        'visible': layer.get('visible'),
                        'opacity': layer.get('opacity'),
                        'adjustment': layer.get('adjustment'),
                    }
                    if (str(record['name']).startswith('PhotoCraft AI') or
                            (isinstance(record['adjustment'], dict) and
                             any('curve' in str(key).lower() for key in record['adjustment']))):
                        report['document_state']['layers'].append(record)
                print('\nRozpoznane warstwy PhotoCraft AI / Curves:',len(report['document_state']['layers']))
                for layer in report['document_state']['layers']:
                    print(json.dumps(layer, ensure_ascii=False, indent=2))
            except Exception as error:
                report['inspect_error'] = str(error)
                print('Nie udało się odczytać doc_inspect:', error)
    path = Path(__file__).with_name('raport_edycja_krzywych_v2_7_5.json')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nRaport:', path)
    print('Nie zmieniono dokumentu ani żadnej warstwy.')


if __name__ == '__main__':
    asyncio.run(main())
