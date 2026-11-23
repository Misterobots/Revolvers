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

## License & Notices

See [LICENSE](LICENSE), [LICENSE-ADDITIONS](LICENSE-ADDITIONS), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
