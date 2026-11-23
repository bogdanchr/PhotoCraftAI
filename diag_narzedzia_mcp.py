# -*- coding: utf-8 -*-
"""Bezpieczna diagnostyka: list_tools() PhotoCraft MCP. Bez wywolan narzedzi i edycji."""
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "http://127.0.0.1:8766/mcp"
BASE_DIR = Path(__file__).resolve().parent
KEYWORDS = ("foreground", "background color", "eyedropper", "pipette", "sample", "color", "state", "inspect", "tool")


async def read_tools():
    records = []
    async with streamable_http_client(MCP_URL) as connection:
        # mcp bywa zwracane jako (read, write) albo (read, write, ...)
        read, write = connection[:2]
        async with ClientSession(read, write) as session:
            await session.initialize()
            cursor = None
            seen_cursors = set()
            while True:
                result = await session.list_tools(cursor=cursor) if cursor else await session.list_tools()
                for tool in result.tools:
                    records.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": tool.inputSchema,
                    })
                cursor = getattr(result, "nextCursor", None)
                if not cursor:
                    break
                if cursor in seen_cursors:
                    raise RuntimeError("Serwer zwrocil powtarzajacy sie kursor stronicowania")
                seen_cursors.add(cursor)
    return records


def main():
    print("PhotoCraft MCP — diagnostyka narzedzi (TYLKO ODCZYT)", flush=True)
    print("Adres:", MCP_URL, flush=True)
    try:
        tools = asyncio.run(read_tools())
        if not tools:
            raise RuntimeError("Serwer nie zwrocil zadnych narzedzi")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = BASE_DIR / "diagnostyka_mcp"
        directory.mkdir(exist_ok=True)
        path = directory / f"narzedzia_mcp_{timestamp}.txt"
        selected = []
        lines = [f"PhotoCraft MCP: {len(tools)} narzedzi", f"Adres: {MCP_URL}", ""]
        for tool in tools:
            blob = (tool['name'] + ' ' + tool['description'] + ' ' + json.dumps(tool['inputSchema'], ensure_ascii=False)).lower()
            if any(word in blob for word in KEYWORDS):
                selected.append(tool)
            lines.append("=" * 70)
            lines.append("NAZWA: " + tool['name'])
            lines.append("OPIS: " + tool['description'])
            lines.append("PARAMETRY: " + json.dumps(tool['inputSchema'], ensure_ascii=False, indent=2))
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8-sig")
        print(f"\nLiczba narzedzi MCP: {len(tools)}")
        print("NAZWY WSZYSTKICH NARZEDZI:")
        for tool in tools:
            print(" - " + tool['name'])
        print(f"\nWyniki zawierajace slowa kluczowe ({len(selected)}):")
        for tool in selected:
            print(" - " + tool['name'])
        print("\nPelny raport (nazwy, opisy i parametry):", path)
        print("Nie wywolano zadnego narzedzia operacyjnego ani nie zmieniono zdjecia.")
        return 0
    except Exception as exc:
        print("\nBLAD:", exc, file=sys.stderr)
        print("Sprawdz uruchomienie PhotoCraft i serwera MCP na porcie 8766.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
