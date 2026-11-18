"""Ostrozna diagnostyka fotografii dla PhotoCraft AI (bez edycji).

Wymaga lokalnego vision_metrics.py (wersja 2, analyze_image).

Przyklady:
    python vision_diagnostics.py oryginal.jpg
    python vision_diagnostics.py oryginal.jpg test.jpg
    python vision_diagnostics.py oryginal.jpg test.jpg --json

Wyniki to sygnaly do sprawdzenia, nie automatyczne diagnozy.
Progi sa wstepnymi ogolnymi heurystykami, nie kalibracja aparatu.
Wersja 1.1: ostrozne wskazowki o zmianach barw, nie o ich poprawnosci.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vision_metrics import analyze_image


def _issue(code: str, severity: str, message: str, evidence: str) -> dict[str, str]:
    return dict(code=code, severity=severity, message=message, evidence=evidence)


def _difference(report: dict[str, Any], key: str) -> float | None:
    values = report.get('bright_neutral_candidates', {}).get('channel_differences')
    if not values:
        return None
    value = values.get(key)
    return float(value) if value is not None else None


def diagnose_single(report: dict[str, Any]) -> dict[str, Any]:
    """Szuka ostroznych sygnalow w pojedynczej fotografii.

    Uwaga: 'bright neutral candidates' NIE sa wzorcem bieli. Syntetyczny
    roz, makijaz, czerwone dekoracje i oswietlenie sceniczne moga dac
    falszywy alarm. Brak sygnalu NIE potwierdza prawidlowych barw.
    """
    issues: list[dict[str, str]] = []
    candidate = report.get('bright_neutral_candidates', {})
    area = float(candidate.get('pct', 0))
    rg = _difference(report, 'R_minus_G')
    bg = _difference(report, 'B_minus_G')
    shadows = report.get('tonal_bands', {}).get('shadows', {}).get('mean_rgb', {})
    shadows_rg = float(shadows['R'] - shadows['G']) if shadows.get('R') is not None and shadows.get('G') is not None else None
    shadows_bg = float(shadows['B'] - shadows['G']) if shadows.get('B') is not None and shadows.get('G') is not None else None

    # Dwa niezalezne sygnaly: jasne, wzglednie neutralne obszary oraz cienie.
    # Celowo konserwatywnie: nie tylko srednia RGB calego kadru.
    if (area >= 5.0 and rg is not None and bg is not None
            and shadows_rg is not None and shadows_bg is not None
            and rg >= 10.0 and bg >= 4.0
            and shadows_rg >= 20.0 and shadows_bg >= 8.0):
        issues.append(_issue(
            'possible_magenta_cast', 'warning',
            'Mozliwa dominanta rozowo-purpurowa; sprawdz neutralne ubrania i skore.',
            f'Jasne kandydaty: {area:.2f}% kadru, R-G={rg:+.2f}, B-G={bg:+.2f}; '
            f'cienie: R-G={shadows_rg:+.2f}, B-G={shadows_bg:+.2f}.',
        ))

    # Niebiesko-fioletowe odchylenie: kandydaci z jasnych miejsc i cienie
    # wskazuja podobny kierunek. INFO, bo kolory moga byc zamierzone.
    # Taki sygnal nie pojawi sie, gdy powierzchnie referencyjne sa zbyt male.
    if (area >= 8.0 and rg is not None and bg is not None
            and shadows_bg is not None
            and bg >= 6.0 and shadows_bg >= 15.0
            and not any(i['code'] == 'possible_magenta_cast' for i in issues)):
        issues.append(_issue(
            'possible_cool_violet_tint', 'info',
            'Mozliwe chlodno-fioletowe przesuniecie jasnych tonow; sprawdz, czy to celowy styl.',
            f'Jasne kandydaty: {area:.2f}% kadru, R-G={rg:+.2f}, B-G={bg:+.2f}; '
            f'cienie: B-G={shadows_bg:+.2f}.',
        ))

    bright = float(report.get('bright_luminance_pct', 0))
    dark = float(report.get('dark_luminance_pct', 0))
    if bright >= 5.0:
        issues.append(_issue(
            'bright_edge', 'info',
            'Znaczny udzial pikseli bardzo jasnych; sprawdz szczegoly swiatel.',
            f'Jasnosc 250-255: {bright:.2f}% kadru (nie musi oznaczac przepalenia).',
        ))
    if dark >= 5.0:
        issues.append(_issue(
            'dark_edge', 'info',
            'Znaczny udzial pikseli bardzo ciemnych; sprawdz szczegoly cieni.',
            f'Jasnosc 0-5: {dark:.2f}% kadru (nie musi oznaczac utraty detali).',
        ))

    return {
        'mode': 'single', 'file': report.get('file'),
        'issues': issues,
        'note': ('Heurystyki oparte na probce obrazu. Kandydaci na neutralne '
                 'powierzchnie nie sa wzorcem bieli. Brak alarmu nie dowodzi '
                 'prawidlowych barw ani kontrastu.'),
    }


def diagnose_pair(original: dict[str, Any], edited: dict[str, Any]) -> dict[str, Any]:
    """Porownanie dwoch wersji zaklada TEN SAM KADR i podobny rozmiar.

    Zmiany pokazuje sie jako opis; nie klasyfikuje ich automatycznie jako blad.
    """
    issues: list[dict[str, str]] = []
    same_size = tuple(original.get('size', ())) == tuple(edited.get('size', ()))
    if not same_size:
        issues.append(_issue(
            'different_dimensions', 'info',
            'Rozne rozdzielczosci: upewnij sie, ze porownujesz ten sam kadr.',
            f"Pierwszy: {original.get('size')}; drugi: {edited.get('size')}",
        ))

    l_change = round(float(edited['mean_luminance']) - float(original['mean_luminance']), 2)
    light_change = round(
        float(edited['tonal_bands']['highlights']['pct']) -
        float(original['tonal_bands']['highlights']['pct']), 2,
    )
    sat_change = round(
        float(edited['saturation']['high_pct']) -
        float(original['saturation']['high_pct']), 2,
    )
    original_rgb = original['mean_rgb']
    edited_rgb = edited['mean_rgb']
    dr = float(edited_rgb['R']) - float(original_rgb['R'])
    dg = float(edited_rgb['G']) - float(original_rgb['G'])
    db = float(edited_rgb['B']) - float(original_rgb['B'])
    # Zmiana relacji kanalow jest bardziej informatywna niz sredni RGB.
    delta_rg = round(dr - dg, 2)
    delta_bg = round(db - dg, 2)

    # Potwierdzaj sygnal na neutralnych kandydatach, ale nie zakladaj,
    # ze zestawy kandydatow zawieraja te same piksele w obu wersjach.
    rg0 = _difference(original, 'R_minus_G')
    rg1 = _difference(edited, 'R_minus_G')
    bg0 = _difference(original, 'B_minus_G')
    bg1 = _difference(edited, 'B_minus_G')
    delta_neutral_rg = round(rg1 - rg0, 2) if rg0 is not None and rg1 is not None else None
    delta_neutral_bg = round(bg1 - bg0, 2) if bg0 is not None and bg1 is not None else None

    if (delta_rg >= 10 and delta_bg >= 10
            and delta_neutral_rg is not None and delta_neutral_bg is not None
            and delta_neutral_rg >= 5 and delta_neutral_bg >= 5):
        issues.append(_issue(
            'magenta_shift', 'warning',
            'Mozliwe przesuniecie kolorystyki w strone purpury miedzy wersjami.',
            f'Zmiana sredniej relacji kanalow: (R-G)={delta_rg:+.2f}, '
            f'(B-G)={delta_bg:+.2f}; jasne kandydaty: '
            f'(R-G)={delta_neutral_rg:+.2f}, (B-G)={delta_neutral_bg:+.2f}.',
        ))
    elif (delta_rg <= -10 and delta_bg <= -10
          and delta_neutral_rg is not None and delta_neutral_bg is not None
          and delta_neutral_rg <= -5 and delta_neutral_bg <= -5):
        issues.append(_issue(
            'green_shift', 'warning',
            'Mozliwe przesuniecie kolorystyki w strone zieleni miedzy wersjami.',
            f'Zmiana relacji kanalow (R-G)={delta_rg:+.2f}, '
            f'(B-G)={delta_bg:+.2f}.',
        ))

    # Dodatkowe, kierunkowe zmiany kolorystyki. Do potwierdzenia wymagamy
    # relacji kanalow calego kadru ORAZ jasnych kandydatow, obecnych w obu
    # wersjach. Nie sa to stale wzorce bieli ani pomiar na tych samych pikselach.
    neutral_a = original.get('bright_neutral_candidates', {})
    neutral_b = edited.get('bright_neutral_candidates', {})
    enough_neutral = (float(neutral_a.get('pct', 0)) >= 5.0
                      and float(neutral_b.get('pct', 0)) >= 5.0)
    if (enough_neutral
            and delta_neutral_bg is not None
            and delta_bg >= 12.0 and delta_neutral_bg >= 6.0
            and delta_rg < 10.0):
        issues.append(_issue(
            'cool_violet_shift', 'info',
            'Wzrosla przewaga niebieskiego nad zielonym; mozliwe ochlodzenie lub fioletowa stylizacja.',
            f'Calosc: zmiana (B-G)={delta_bg:+.2f}; jasne kandydaty: '
            f'zmiana (B-G)={delta_neutral_bg:+.2f} pkt RGB. '
            'To nie dowod nieprawidlowej kolorystyki.',
        ))
    if (enough_neutral
            and delta_neutral_rg is not None
            and delta_rg >= 8.0 and delta_neutral_rg >= 5.0
            and delta_bg < 6.0
            and not any(i['code'] == 'magenta_shift' for i in issues)):
        issues.append(_issue(
            'warm_red_shift', 'info',
            'Wzrosla przewaga czerwieni nad zielenia; sprawdz odcienie skory i neutralne przedmioty.',
            f'Calosc: zmiana (R-G)={delta_rg:+.2f}; jasne kandydaty: '
            f'zmiana (R-G)={delta_neutral_rg:+.2f} pkt RGB. '
            'Mozliwa zamierzona ciepla stylizacja.',
        ))

    if abs(l_change) >= 10 or abs(light_change) >= 8:
        issues.append(_issue(
            'tonality_shift', 'info',
            'Wyrazna zmiana tonalnosci miedzy wersjami; sprawdz, czy zamierzona.',
            f'Srednia jasnosc {l_change:+.2f}/255; '
            f'udzial swiatel {light_change:+.2f} pkt proc.',
        ))
    if abs(sat_change) >= 5:
        issues.append(_issue(
            'saturation_shift', 'info',
            'Zmienil sie udzial mocno nasyconych pikseli.',
            f'Zmiana: {sat_change:+.2f} pkt proc.',
        ))

    return {
        'mode': 'comparison',
        'original': original.get('file'), 'edited': edited.get('file'),
        'issues': issues,
        'deltas': {
            'mean_luminance': l_change,
            'highlight_share_pp': light_change,
            'high_saturation_share_pp': sat_change,
            'mean_r_minus_g': delta_rg,
            'mean_b_minus_g': delta_bg,
            'candidate_r_minus_g': delta_neutral_rg,
            'candidate_b_minus_g': delta_neutral_bg,
        },
        'note': ('Wynik porownania dotyczy tego samego kadru. Kandydaci na '
                 'neutralne obszary sa wybierani osobno w kazdej wersji. '
                 'Zauwazona zmiana nie jest automatycznie bledem.'),
    }


def print_diagnostics(diagnosis: dict[str, Any]) -> None:
    mode = diagnosis['mode']
    print('\nDIAGNOSTYKA:', 'POROWNANIE' if mode == 'comparison' else 'JEDEN PLIK')
    if not diagnosis['issues']:
        print('Brak silnych sygnalow wedlug aktualnych heurystyk.')
    for finding in diagnosis['issues']:
        print(f"[{finding['severity'].upper()}] {finding['code']}: {finding['message']}")
        print(f"  Dane: {finding['evidence']}")
    print('UWAGA:', diagnosis['note'])


def main() -> None:
    parser = argparse.ArgumentParser(description='Odczytowa diagnostyka fotografii')
    parser.add_argument('image', type=Path, help='Zdjecie lub oryginal')
    parser.add_argument('second_image', nargs='?', type=Path, help='Opcjonalnie wersja po korekcie')
    parser.add_argument('--json', action='store_true', help='Dane do przyszlej integracji')
    args = parser.parse_args()
    original = analyze_image(args.image)
    if args.second_image is not None:
        edited = analyze_image(args.second_image)
        report = diagnose_pair(original, edited)
    else:
        report = diagnose_single(original)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_diagnostics(report)


if __name__ == '__main__':
    main()
