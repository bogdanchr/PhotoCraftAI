"""Odczytowe pomiary fotografii dla PhotoCraft AI (wersja 2).

Uzycie:
    python vision_metrics.py zdjecie.jpg
    python vision_metrics.py oryginal.jpg po_korekcie.jpg

Wymaga Pillow. Nie edytuje plikow. Zachowuje interfejs analyze_image()
wykorzystywany przez app.py (klucze mean_rgb, size itd.).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageStat, ImageOps

MAX_SIDE = 1600


def _percentile(histogram: list[int], percentile: float) -> int:
    count = sum(histogram)
    if not count:
        return 0
    target = count * percentile / 100.0
    cumulative = 0
    for level, frequency in enumerate(histogram):
        cumulative += frequency
        if cumulative >= target:
            return level
    return 255


def _percent(histogram: list[int], first: int, last: int) -> float:
    count = sum(histogram)
    return 100.0 * sum(histogram[first:last + 1]) / count if count else 0.0


def _round_rgb(sums: list[int], count: int) -> dict:
    return {name: round(v / count, 2) if count else None
            for name, v in zip(('R', 'G', 'B'), sums)}


def analyze_image(path: str | Path) -> dict:
    """Przyblizone statystyki sRGB (probka maks. 1600 px na bok).

    Progi i przedzialy sa diagnostyczne, a NIE kryteriami poprawnosci
    balansu bieli. Z danych pojedynczego zdjecia nie da sie ustalic,
    czy neutralny fragment sceny powinien byc bialy lub szary.
    """
    path = Path(path)
    with Image.open(path) as opened:
        size = opened.size
        img = ImageOps.exif_transpose(opened).convert('RGB')
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    rgb = ImageStat.Stat(img).mean
    rgb_hist = img.histogram()
    luminance = img.convert('L')
    lum_hist = luminance.histogram()
    mean_lum = ImageStat.Stat(luminance).mean[0]
    channel_names = ('R', 'G', 'B')
    clipping = {
        name: {
            'near_0_pct': round(_percent(rgb_hist[i * 256:(i + 1) * 256], 0, 5), 2),
            'near_255_pct': round(_percent(rgb_hist[i * 256:(i + 1) * 256], 250, 255), 2),
        }
        for i, name in enumerate(channel_names)
    }

    # Statystyki wg jasnosci: cienie 0..84, poltony 85..169,
    # swiatla 170..255. Przedzialy sa stale, niezalezne od kadru.
    tonal_names = ('shadows', 'midtones', 'highlights')
    tonal = {name: {'count': 0, 'sums': [0, 0, 0]} for name in tonal_names}
    sat_hist = [0] * 256
    near_neutral = {'count': 0, 'sums': [0, 0, 0]}
    n = img.width * img.height
    # Ta para iteratorow opisuje dokladnie te same piksele w tej samej kolejnosci.
    for (r, g, b), lum in zip(img.get_flattened_data(), luminance.get_flattened_data()):
        band = tonal_names[min(lum // 85, 2)]
        acc = tonal[band]
        acc['count'] += 1
        acc['sums'][0] += r
        acc['sums'][1] += g
        acc['sums'][2] += b

        maximum, minimum = max(r, g, b), min(r, g, b)
        sat = round((maximum - minimum) / maximum * 255) if maximum else 0
        sat_hist[sat] += 1
        # Kandydaci na JASNE, WZGLEDNIE neutralne powierzchnie.
        # Prog toleruje umiarkowany zafarb, ale moze obejmowac szare,
        # jasnobrazowe, rozowe i inne przedmioty. To nie pomiar balansu bieli.
        if lum >= 160 and (maximum - minimum) <= 45:
            near_neutral['count'] += 1
            near_neutral['sums'][0] += r
            near_neutral['sums'][1] += g
            near_neutral['sums'][2] += b

    tonal_bands = {
        name: {
            'pct': round(100 * v['count'] / n, 2) if n else 0.0,
            'mean_rgb': _round_rgb(v['sums'], v['count'])
        }
        for name, v in tonal.items()
    }
    neutral_count = near_neutral['count']
    neutral_mean = _round_rgb(near_neutral['sums'], neutral_count)
    neutral_differences = (
        {'R_minus_G': round(neutral_mean['R'] - neutral_mean['G'], 2),
         'B_minus_G': round(neutral_mean['B'] - neutral_mean['G'], 2)}
        if neutral_count else None
    )

    return {
        'file': str(path),
        'size': size,
        'sample_size': img.size,
        'mean_rgb': {name: round(value, 2) for name, value in zip(channel_names, rgb)},
        'mean_luminance': round(mean_lum, 2),
        'luminance_p01': _percentile(lum_hist, 1),
        'luminance_p50': _percentile(lum_hist, 50),
        'luminance_p99': _percentile(lum_hist, 99),
        'dark_luminance_pct': round(_percent(lum_hist, 0, 5), 2),
        'bright_luminance_pct': round(_percent(lum_hist, 250, 255), 2),
        'channels': clipping,
        'tonal_bands': tonal_bands,
        'saturation': {
            'p50': _percentile(sat_hist, 50),
            'p90': _percentile(sat_hist, 90),
            'high_pct': round(_percent(sat_hist, 204, 255), 2),
        },
        'bright_neutral_candidates': {
            'pct': round(100 * neutral_count / n, 2) if n else 0.0,
            'mean_rgb': neutral_mean,
            'channel_differences': neutral_differences,
            'rule': 'luminance >= 160 and max(RGB)-min(RGB) <= 45',
        },
    }


def print_report(report: dict) -> None:
    print(f"\nPLIK: {report['file']}")
    print(f"Rozdzielczosc: {report['size'][0]} x {report['size'][1]}")
    rgb = report['mean_rgb']
    print(f"Srednie RGB: R={rgb['R']:.2f} G={rgb['G']:.2f} B={rgb['B']:.2f}")
    print(f"Srednia jasnosc 0-255: {report['mean_luminance']:.2f}")
    print('Jasnosc (percentyle 1/50/99): '
          f"{report['luminance_p01']} / {report['luminance_p50']} / {report['luminance_p99']}")
    print('Prawie czarna/biala jasnosc (<=5 / >=250): '
          f"{report['dark_luminance_pct']:.2f}% / {report['bright_luminance_pct']:.2f}%")
    for name, values in report['channels'].items():
        print(f"Kanal {name} prawie 0 / prawie 255: "
              f"{values['near_0_pct']:.2f}% / {values['near_255_pct']:.2f}%")
    print('\nSTREFY TONALNE (udzial oraz srednie RGB):')
    for name, values in report['tonal_bands'].items():
        print(f"  {name}: {values['pct']:.2f}%; RGB={values['mean_rgb']}")
    sat = report['saturation']
    print(f"NASYCENIE (HSV, 0-255): mediana={sat['p50']}, p90={sat['p90']}, "
          f"mocno nasycone >=204: {sat['high_pct']:.2f}%")
    candidate = report['bright_neutral_candidates']
    print('JASNE WZGLEDNIE NEUTRALNE OBSZARY (kandydaci, nie wzorzec bieli):')
    print(f"  udzial={candidate['pct']:.2f}%; RGB={candidate['mean_rgb']}; "
          f"roznice={candidate['channel_differences']}")
    print('UWAGA: te statystyki nie ustalaja same, czy zafarb jest bledem.')


def print_comparison(original: dict, edited: dict) -> None:
    print('\nROZNICA: DRUGI PLIK MINUS PIERWSZY')
    for name in ('R', 'G', 'B'):
        delta = edited['mean_rgb'][name] - original['mean_rgb'][name]
        print(f'Srednia {name}: {delta:+.2f}')
    delta_l = edited['mean_luminance'] - original['mean_luminance']
    print(f'Srednia jasnosc: {delta_l:+.2f}')
    print('UWAGA: Srednie kanaly RGB nie dowodza same w sobie zafarbu.')
    print('Porownanie jest najbardziej sensowne dla tego samego kadru przed i po edycji.')
    for name in ('shadows', 'midtones', 'highlights'):
        a = original['tonal_bands'][name]['pct']
        b = edited['tonal_bands'][name]['pct']
        print(f'Strefa {name}: {b-a:+.2f} pkt proc.')
    a, b = original['saturation'], edited['saturation']
    print(f"Nasycenie mocne >=204: {b['high_pct'] - a['high_pct']:+.2f} pkt proc.")
    print('Jasne wzgl. neutralne obszary sa selekcjonowane niezaleznie w obu plikach.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Pomiary fotografii bez zmian plikow')
    parser.add_argument('image', help='Sciezka do obrazu')
    parser.add_argument('second_image', nargs='?', help='Opcjonalnie obraz po korekcie')
    args = parser.parse_args()
    first = analyze_image(args.image)
    print_report(first)
    if args.second_image:
        second = analyze_image(args.second_image)
        print_report(second)
        print_comparison(first, second)


if __name__ == '__main__':
    main()
