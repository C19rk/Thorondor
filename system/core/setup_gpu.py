#!/usr/bin/env python3
"""
setup_gpu.py — One-click GPU setup for Argus
=============================================
Run this once from the project root, then restart app.py or wcapp.py.

What it does:
  - Detects your GPU (NVIDIA / AMD / Intel)
  - Installs the correct ONNX Runtime package for GPU acceleration
  - NVIDIA: keeps standard onnxruntime (CUDA is auto-detected)
  - AMD / Intel on Windows: swaps onnxruntime → onnxruntime-directml

Usage:
    python setup_gpu.py
"""

import sys
import subprocess
import platform


def run(cmd: list[str]) -> int:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def pip(*args):
    return run([sys.executable, "-m", "pip"] + list(args))


def sep(char="─", w=56):
    print(char * w)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sep()
    print("  Argus GPU Setup")
    sep()
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print()

    # ── NVIDIA / CUDA ─────────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  NVIDIA GPU detected: {name}")
            print("  Standard onnxruntime already supports CUDA — nothing to install.")
            sep()
            print("  Done! Restart the app.")
            return
    except ImportError:
        pass

    # ── Check current ORT providers ───────────────────────────────────────────
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  Current ORT providers: {providers}")
    except ImportError:
        providers = []

    if "DmlExecutionProvider" in providers:
        print("  DirectML already active — AMD/Intel GPU acceleration is enabled.")
        sep()
        print("  Done! Restart the app to apply.")
        return

    # ── AMD / Intel on Windows → install DirectML ────────────────────────────
    if sys.platform != "win32":
        print("  No CUDA GPU found and this system is not Windows.")
        print("  Only CUDA (NVIDIA) and DirectML (Windows AMD/Intel) are supported.")
        print("  The app will run on CPU.")
        return

    print()
    print("  AMD/Intel GPU detected on Windows.")
    print("  Installing onnxruntime-directml (replaces standard onnxruntime)...")
    print()

    rc1 = pip("uninstall", "onnxruntime", "-y")
    rc2 = pip("install", "onnxruntime-directml")

    print()
    if rc2 == 0:
        sep()
        print("  Install complete!")
        print()
        print("  Verifying DirectML provider...")
        # Re-import to check
        try:
            import importlib
            ort_mod = importlib.import_module("onnxruntime")
            importlib.reload(ort_mod)
            p2 = ort_mod.get_available_providers()
            if "DmlExecutionProvider" in p2:
                print("  DirectML is active.")
            else:
                print("  ! DirectML not yet showing — please restart your terminal and try again.")
        except Exception:
            print("  Restart your terminal to verify.")
        sep()
        print("  Restart the app: python app.py  or  python wcapp.py")
    else:
        sep("!", 56)
        print("  Install failed. Try manually:")
        print("      pip uninstall onnxruntime -y")
        print("      pip install onnxruntime-directml")
        sep("!", 56)


if __name__ == "__main__":
    main()