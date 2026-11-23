# -*- coding: utf-8 -*-
"""PhotoCraft AI: audyt ETAPU 1 z istniejacego katalogu MCP.
Odczyt lokalnego polecenia_photocraft.txt. Zero polaczen MCP, zero edycji obrazu.
Uruchom w D:\\PhotoCraftAI: python audyt_korekcji_swiatla_i_koloru.py
"""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'polecenia_photocraft.txt'
REPORT = ROOT / 'raport_etap1_korekcja_swiatla_koloru.txt'
JSON_REPORT = ROOT / 'raport_etap1_korekcja_swiatla_koloru.json'
TOPICS = {
    'Krzywe (Curves)': [r'curves?', r'krzyw'],
    'Poziomy (Levels)': [r'levels?', r'poziom'],
    'Jasność / Kontrast': [r'brightness', r'contrast', r'jasno', r'kontrast'],
    'Ekspozycja (Exposure)': [r'exposure', r'ekspozycj'],
    'Cienie / Światła': [r'shadows?', r'highlights?', r'cieni', r'światł', r'swiatl'],
    'Balans kolorów (Color Balance)': [r'color.?balance', r'colour.?balance', r'balans.?kolor'],
    'Kolor selektywny (Selective Color)': [r'selective.?colou?r', r'selektywn'],
    'Barwa / Nasycenie (Hue/Saturation)': [r'hue.?saturation', r'hue', r'saturation', r'nasycen', r'barwa'],
}


def catalog_from_text(raw):
    raw = raw.strip().lstrip('\ufeff')
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"TextContent\(type=['\"]text['\"],\s*text=(['\"])", raw)
        if not match:
            raise ValueError('Nie rozpoznano struktury katalogu JSON/TextContent. Nie zmieniono pliku.')
        # Literal w starszym zrzucie; pobieramy tylko string tekstowy JSON.
        import io, tokenize
        sub = raw[match.start():]
        prefix = re.search(r'\btext\s*=\s*', sub)
        seq = tokenize.generate_tokens(io.StringIO(sub[prefix.end():]).readline)
        token = next((tok for tok in seq if tok.type == tokenize.STRING), None)
        if token is None:
            raise ValueError('Nie udalo sie wydobyc katalogu TextContent')
        obj = json.loads(ast.literal_eval(token.string))
    if isinstance(obj, dict):
        obj = obj.get('commands', obj.get('result', obj.get('data')))
    if not isinstance(obj, list) or not all(isinstance(x, dict) and isinstance(x.get('id'), str) for x in obj):
        raise ValueError('Nieprawidlowa lista komend')
    return obj


def main():
    if not SOURCE.exists():
        raise SystemExit('Brak pliku polecenia_photocraft.txt w folderze skryptu. Uruchom w D:\\PhotoCraftAI.')
    commands = catalog_from_text(SOURCE.read_text(encoding='utf-8-sig'))
    sections = {}
    lines = ['PHOTOCRAFT AI — ETAP 1: KOREKCJA SWIATLA I KOLORU',
             'Audyt odczytowy z istniejacego pliku polecenia_photocraft.txt',
             f'Komend w katalogu: {len(commands)}',
             'UWAGA: Dopasowanie nazwy nie dowodzi, ze komenda dziala ani ze jest niedestrukcyjna.',
             'Stan enabled jest chwilowy i moze zalezec od aktywnej warstwy.', '']
    for topic, patterns in TOPICS.items():
        matches = []
        for command in commands:
            # Dopasowujemy ID i etykiete, lecz nie rozbudowany opis params:
            # ten zwykle zawiera nazwy wielu innych korekt i daje falszywe trafienia.
            text = ' '.join(str(command.get(k, '')) for k in ('id', 'label'))
            if any(re.search(p, text, re.IGNORECASE) for p in patterns):
                matches.append({k: command.get(k) for k in ('id','label','enabled','disabledReason','params','menu')})
        matches.sort(key=lambda x: x['id'])
        sections[topic] = matches
        lines.extend([f'=== {topic} ===', f'Dopasowanych komend: {len(matches)}'])
        if not matches:
            lines.append('BRAK bezposredniego dopasowania nazwy — wymaga dalszego sprawdzenia.')
        for item in matches:
            lines.append(f"- {item['id']} | {item.get('label')} | enabled={item.get('enabled')}")
            lines.append('  params: ' + str(item.get('params') or '(brak)'))
            if item.get('disabledReason'):
                lines.append('  disabledReason: ' + str(item['disabledReason']))
        lines.append('')
    output = {'source': SOURCE.name, 'command_count': len(commands), 'read_only': True,
              'sections': sections, 'note': 'Dopasowania do weryfikacji; nie stanowia testu wykonania.'}
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    JSON_REPORT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n'.join(lines[:5]))
    for name, matches in sections.items():
        print(f'{name}: {len(matches)} komend(y)')
    print('\nZapisano:', REPORT, '\noraz:', JSON_REPORT)
    print('Nie polaczono sie z MCP i nie zmieniono zadnego zdjecia.')

if __name__ == '__main__':
    main()
