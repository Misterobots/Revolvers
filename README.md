# Revolvers

A clean, **build-from-dump** Windows installer and build kit for running **Orbitals** natively on PC using your own legally dumped Switch 2 copy.

> **Disclaimer:** This repository and its releases contain **zero** copyrighted game code, game assets, binaries, or shader caches. You **must** supply your own original Switch 2 copy of *Orbitals* (in `.nup` or extracted `exefs/` + `romfs/` folder format) during installation.

---

## Quick Start (Recommended)

1. Download **`Orbitals-Builder-Setup.exe`** from the [Releases](../../releases) page.
2. Run **`Orbitals-Builder-Setup.exe`** and choose your install directory (defaults to `C:\Games\Orbitals`).
3. Select your Switch 2 dump:
   - **`.NUP` file** *(recommended)*, or
   - **Extracted game folder** containing `exefs/` and `romfs/`.
4. Wait ~2–5 minutes while the installer verifies the game ID, lifts and compiles `orbitals.exe`, extracts `romfs`, and creates Start Menu / Desktop shortcuts.

### In-Game Menu & Controls
- **Open Menu:** `Esc` or `F1` on keyboard, or **Guide/Home** / **`Back+Start` (`Minus+Plus`)** on any connected controller.
- **Controls tab:** Assign Player 1 and Player 2 independently (`Auto`, `Gamepad 1`, `Gamepad 2`, `Keyboard`, or `Off` for P2).
- **Screen tab:** Supports Bilinear & AMD FSR 1 upscaling, FPS limit (`30` / `60` / `Unlimited`), and Presentation / VSync modes.
- **Saves & Settings:** Stored locally in `profile/` and `nx2navi.ini` inside your install folder (preserved automatically on reinstall or uninstall).

---

## System Requirements

- **OS:** Windows 10 / 11 (64-bit)
- **GPU:** Vulkan-capable GPU with current drivers (NVIDIA / AMD / Intel)
- **Disk Space:** ~15 GB free during build (~11 GB after temporary build cache is cleaned up)
- **Build Prerequisites (auto-installed if missing during setup):**
  - Python 3 (`numpy`, `lz4`, `zstandard`, `zbic`)
  - MSYS2 UCRT64 (`clang`, `gcc`, `sdl3`, `opus`, `libvpx`, `dav1d`, `zstd`, `vulkan-headers`, `vulkan-loader`, `vulkan-memory-allocator`)
  - .NET 10 Runtime (for shader translation of new shaders)

---

## Building the Installer From Source

1. Install [Inno Setup 6](https://jrsoftware.org/isinfo.php).
2. Compile `orbitals-builder.iss`:
   ```powershell
   & "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" orbitals-builder.iss
   ```
3. The compiled installer is written to `dist\Orbitals-Builder-Setup.exe`.

---

## RecompLauncher

Revolvers ships a `stage/recomplauncher.json` manifest for [RecompLauncher](https://recomplauncher.com/sdk). The player picks their own Orbitals Digital Deluxe `.nup`; the launcher's one-time setup (`kit/tools/recomplauncher_prepare.py`) builds the game from it, the same as the installer, and then prepares the shaders so the first play does not stop to translate them.

**Shader preparation.** Orbitals keeps its shaders in its Unreal Engine 5 IoStore container, Oodle-compressed. `kit/tools/ue_shaders.py` reads them from the player's own files, using the game's own decoder through `orbitals.exe --oodle-serve` (`kit/tools/oodle_find.py` locates it in `main`). It joins them with the shader list in `kit/tools/data/orbitals/`, which holds code hashes, bind states and pipeline states recorded in play and no game code. Then it runs `--prepare-shaders` and `--prewarm`, which build every recorded shader and pipeline into the caches.

**Updating the shader list** from a build that has been played further (do this before a release):

```powershell
python stage\kit\tools\ue_shaders.py <build> --strip stage\kit\tools\data\orbitals\shader-inputs.bin
copy <build>\shader-answers.bin stage\kit\tools\data\orbitals\
copy <build>\pipeline-keys.bin stage\kit\tools\data\orbitals\
```

**Release zip** (`Orbitals-v<version>-Windows-x64.zip`): `recomplauncher.json`, `nx2navi.ini` and the contents of `sidecars/` at the top, and `kit/` without `*.bak*` files. Never include game files. Setup runs with the translator settings in the manifest's `prepare.env`, which must match `start.env`, because the shader cache is keyed by them.
## License & Notices

See [LICENSE](LICENSE), [LICENSE-ADDITIONS](LICENSE-ADDITIONS), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
