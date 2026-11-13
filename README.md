# PhotoCraft AI

**A community-oriented, local AI integration layer for the PhotoCraft image editor.**

PhotoCraft AI connects the external PhotoCraft editor with local language and vision models, ComfyUI workflows, and an MCP-based control interface. It aims to make photographic analysis, object masking, and selected editing operations accessible from a single Gradio interface.

> **Project status:** Work in progress. The repository contains integration scripts and configuration examples, **not** the PhotoCraft editor, AI model weights, or the external applications required to run them. Installation has so far been developed and tested in a Windows-based local setup; it is not yet a one-click installer.

**[Wersja polska / Polish README](README_PL.md)**

## Features

- **General AI chat:** a Qwen-powered Gradio conversation interface for questions and PhotoCraft-related tasks.
- **Photo analysis:** Qwen3-VL can review an exported photograph and suggest improvements without automatically modifying the image.
- **BiRefNet:** background/object extraction, including a workflow for processing a selected area.
- **SAM3:** text-prompt-based object segmentation and mask workflows through ComfyUI.
- **Face adjustments:** assisted brightness/contrast and mask workflows, with manual refinement available in the editor.
- **MCP integration:** communication with PhotoCraft's automation/control interface, using a local command catalog.
- **Windows launcher and shortcut:** scripts to start local services and optionally configure a BiRefNet keyboard shortcut.

Some operations modify the open document. Review results in PhotoCraft and save your work appropriately.

## Architecture

```text
Gradio UI (app.py)
  |-- Qwen / Qwen3-VL via Ollama
  |-- PhotoCraft control via MCP
  |-- BiRefNet and SAM3 bridges --> ComfyUI
  `-- Local import/export directories <--> PhotoCraft editor
```

PhotoCraft AI is an **independent integration project**. PhotoCraft, ComfyUI, Ollama, the AI models, and their respective licenses are separate from this repository.

## Requirements

- Windows (the included launchers use `.bat` and PowerShell).
- Python and a virtual environment for this project.
- PhotoCraft editor with the compatible local control interface.
- ComfyUI with the required BiRefNet/SAM3 workflows, nodes, and models.
- Ollama with suitable text and vision models (the current application configuration references `qwen3:14b` and `qwen3-vl:8b`; adjust to your installation).
- A separate Python environment providing `mcp-proxy` for the launcher configuration.

A suitable GPU is recommended for local AI processing. Exact model/node versions and hardware requirements still need to be documented and verified.

## Repository structure

```text
PhotoCraftAI/
├── app.py                         # Gradio application
├── PhotoCraft_AI.bat              # Windows service launcher
├── local_config.example.bat       # Configuration template
├── requirements.txt              # Python dependencies
├── polecenia_photocraft.txt       # PhotoCraft command catalog
├── comfy_bridge/                 # ComfyUI / PhotoCraft bridge scripts and API workflows
├── shortcuts/                    # BiRefNet shortcut script
└── PhotoCraft_BiRefNet_CtrlAltB/  # Optional shortcut installer and instructions
```

## Initial setup (Windows)

These are **preliminary setup steps**, not a verified clean-machine installation guide.

1. Install PhotoCraft, ComfyUI, Ollama, and the necessary models/nodes separately, according to their own documentation.
2. Clone or download this repository and open a terminal in its root directory.
3. Create a Python virtual environment and install the dependencies:

   ```powershell
   py -m venv venv
   .\venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Copy `local_config.example.bat` to `local_config.bat` and adapt the paths to your computer. **Do not commit `local_config.bat` or place passwords/tokens in it.**
5. Check the launcher assumptions and ensure that the PhotoCraft control token is stored outside the repository in the location expected by your configuration.
6. Start `PhotoCraft_AI.bat` only after the external components and required Python environments are ready.

The launcher currently expects a specific local installation layout and service ports; further portability work and a full dependency checklist are planned.

## Configuration and local data

- `local_config.example.bat` is a public template; `local_config.bat` is machine-specific and excluded from Git.
- `PHOTOCRAFT_IMPORT_DIR` and `PHOTOCRAFT_EXPORT_DIR` can control local exchange folders.
- `PHOTOCRAFT_COMFY_PYTHON` can point bridge scripts to the Python interpreter bundled with ComfyUI.
- Keep model files, exports, temporary images, logs, credentials, and tokens out of version control.
- The PhotoCraft command catalog is consumed locally by the Gradio application; it is not a substitute for a compatible PhotoCraft control interface.

## Known limitations

- Primarily developed on Windows; other operating systems are not currently documented or supported by the supplied launch scripts.
- External software, workflows, nodes, and model downloads are not bundled.
- Some automation actions require visual confirmation; model output is not a guarantee of correct edits.
- Fresh-install instructions, dependency pinning, compatibility matrix, and automated tests are still being developed.

## Contributing

Feedback, bug reports, documentation improvements, and pull requests are welcome once the repository is published. Please avoid submitting private photos, personal file paths, credentials, tokens, or proprietary model files in issues and pull requests.


## License and attribution

PhotoCraft AI is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

**Copyright (c) 2026 Bogdan Chrobak**

The MIT License applies to the PhotoCraft AI integration code. The PhotoCraft editor, ComfyUI, Ollama, and external AI models are separate projects governed by their own licenses and terms.