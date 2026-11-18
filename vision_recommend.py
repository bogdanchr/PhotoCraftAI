import re

def recommend_photocraft_tools(analysis):
    match = re.search(
        r"3\.\s*POPRAWKI\s*:?(.*?)(?=4\.|$)",
        analysis,
        re.I | re.S
    )

    if not match:
        return "Brak sekcji POPRAWKI."

    corrections = match.group(1).strip()

    negative = [
        "nie wymaga",
        "brak potrzeby",
        "nie ma potrzeby",
        "bez konieczności",
        "nie proponuję",
    ]

    if any(x in corrections.lower() for x in negative):
        return "Nie wykryto koniecznych korekt."

    suggestions = []

    if re.search(r"rozjaśni|przyciemni|jasnoś", corrections, re.I):
        suggestions.append(
            "Jasność: możliwa warstwa Jasność/Kontrast; "
            "MCP wymaga zatwierdzonych wartości."
        )

    if re.search(r"zwiększ.*kontrast|zmniejsz.*kontrast|popraw.*kontrast", corrections, re.I):
        suggestions.append(
            "Kontrast: dostępna korekta Jasność/Kontrast; "
            "MCP wymaga zatwierdzonych wartości."
        )

    if re.search(r"\bpoziomy\b|\bpoziomach\b|punkt czerni|punkt bieli|gamma", corrections, re.I):
        suggestions.append(
            "Poziomy: narzędzie Levels dostępne w PhotoCraft; "
            "należy ustalić parametry i sposób wykonania."
        )

    if not suggestions:
        return "Brak jednoznacznego dopasowania narzędzi."

    return "\n".join(suggestions)


if __name__ == "__main__":
    tests = [
        "3. POPRAWKI:\nWarto delikatnie rozjaśnić twarz.\n4. PODSUMOWANIE:",
        "3. POPRAWKI:\nZdjęcie nie wymaga istotnych zmian.\n4. PODSUMOWANIE:",
        "3. POPRAWKI:\nMożna skorygować punkt czerni w poziomach.\n4. PODSUMOWANIE:",
    ]

    for i, test in enumerate(tests, 1):
        print(f"\nTEST {i}")
        print(recommend_photocraft_tools(test))


def find_catalog_commands(catalog):
    wanted = {
        "layer.newAdjustmentLayer.brightnessContrast",
        "layer.newAdjustmentLayer.levels",
    }
    return {
        item["id"]: item
        for item in catalog
        if isinstance(item, dict) and item.get("id") in wanted
    }



def match_recommendations_to_catalog(analysis, catalog):
    recommendations = recommend_photocraft_tools(analysis)

    if recommendations.startswith(("Brak", "Nie wykryto")):
        return recommendations

    commands = find_catalog_commands(catalog)
    lines = []

    mappings = [
        ("brightness", "layer.newAdjustmentLayer.brightnessContrast"),
        ("contrast", "layer.newAdjustmentLayer.brightnessContrast"),
        ("levels", "layer.newAdjustmentLayer.levels"),
    ]

    text = recommendations.casefold()

    for kind, command_id in mappings:
        if kind == "brightness":
            matched = "jasno" in text
        elif kind == "contrast":
            matched = "kontrast:" in text
        else:
            matched = "poziomy:" in text

        if not matched:
            continue

        command = commands.get(command_id)

        if command is None:
            lines.append(f"{kind}: not found in catalog")
            continue

        status = "enabled" if command.get("enabled") else "disabled"
        lines.append(
            f"{kind}: {command_id} ({status}). "
            "Wykonanie wymaga potwierdzenia parametrow i zgody uzytkownika."
        )

    return "\\n".join(lines) if lines else "Brak dopasowanych polecen."
