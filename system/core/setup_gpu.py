#!/usr/bin/env python3
"""
setup_gpu.py — One-click GPU setup for Argus
=============================================
Run this ONCE from the system/ folder, then restart app.py or wcapp.py.

What it does:
  NVIDIA GPU  : Verifies CUDA is available via PyTorch. No extra install needed.
  AMD GPU     : Linux  → Checks for ROCm PyTorch. Prints install command if missing.
                Windows → Installs onnxruntime-directml for DirectML support.
  Intel GPU   : Windows → Same DirectML install as AMD.
  No GPU      : Informs you the app will run on CPU (still works fine).

Usage:
    python setup_gpu.py
"""

import sys
import subprocess
import platform


def run(cmd: list) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def pip(*args):
    return run([sys.executable, "-m", "pip"] + list(args))


def sep(char="─", w=60):
    print(char * w)


def _is_rocm(torch) -> bool:
    try:
        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def main():
    sep()
    print("  Argus GPU Setup")
    sep()
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print()

    # ── NVIDIA CUDA ───────────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available() and not _is_rocm(torch):
            name = torch.cuda.get_device_name(0)
            print(f"  NVIDIA GPU detected: {name}")
            print("  Standard onnxruntime already supports CUDA — nothing to install.")
            sep()
            print("  Done! Restart the app to apply GPU acceleration.")
            return
    except ImportError:
        pass

    # ── AMD ROCm (Linux) ──────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available() and _is_rocm(torch):
            name = torch.cuda.get_device_name(0)
            print(f"  AMD GPU detected (ROCm): {name}")
            try:
                import onnxruntime as ort
                if "ROCMExecutionProvider" in ort.get_available_providers():
                    print("  ROCMExecutionProvider already active — AMD GPU acceleration is enabled.")
                    sep()
                    print("  Done! Restart the app to apply GPU acceleration.")
                    return
            except ImportError:
                pass
            print()
            print("  onnxruntime-rocm is not installed. Installing...")
            print()
            rc = pip("install", "onnxruntime-rocm")
            if rc == 0:
                sep()
                print("  Install complete! Restart the app.")
            else:
                sep("!", 60)
                print("  Install failed. Try manually:")
                print("      pip install onnxruntime-rocm")
                print()
                print("  Or visit: https://onnxruntime.ai/docs/install/")
                sep("!", 60)
            return
    except ImportError:
        pass

    # ── AMD on Linux but no ROCm PyTorch installed ────────────────────────────
    if sys.platform != "win32":
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5
            )
            if "AMD" in result.stdout or "Radeon" in result.stdout or "ATI" in result.stdout:
                print("  AMD GPU detected on Linux — but ROCm PyTorch is NOT installed.")
                print()
                print("  Install the ROCm version of PyTorch, then re-run this script:")
                print("  https://pytorch.org/get-started/locally/  (select ROCm)")
                print()
                print("  Example (ROCm 6.x):")
                print("      pip install torch --index-url https://download.pytorch.org/whl/rocm6.0")
                sep()
                return
        except Exception:
            pass
        print("  No GPU detected (or GPU drivers are not installed).")
        print("  The app will run on CPU — slower but still works.")
        sep()
        return

    # ── AMD / Intel on Windows → DirectML ────────────────────────────────────
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  Current ORT providers: {providers}")
        if "DmlExecutionProvider" in providers:
            print("  DirectML already active — AMD/Intel GPU acceleration is enabled.")
            sep()
            print("  Done! Restart the app to apply GPU acceleration.")
            return
    except ImportError:
        providers = []

    print()
    print("  AMD/Intel GPU on Windows detected.")
    print("  Installing onnxruntime-directml (replaces standard onnxruntime)...")
    print()

    pip("uninstall", "onnxruntime", "-y")
    rc = pip("install", "onnxruntime-directml")

    print()
    if rc == 0:
        sep()
        print("  Install complete!")
        print()
        print("  Verifying DirectML provider...")
        try:
            import importlib
            import onnxruntime as ort_mod
            importlib.reload(ort_mod)
            if "DmlExecutionProvider" in ort_mod.get_available_providers():
                print("  DirectML is active.")
            else:
                print("  ! DirectML not showing yet — please restart your terminal.")
        except Exception:
            print("  Restart your terminal to verify.")
        sep()
        print("  Restart the app: python app.py  or  python wcapp.py")
    else:
        sep("!", 60)
        print("  Install failed. Try manually:")
        print("      pip uninstall onnxruntime -y")
        print("      pip install onnxruntime-directml")
        sep("!", 60)


if __name__ == "__main__":
    main()