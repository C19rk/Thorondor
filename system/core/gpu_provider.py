"""
core/gpu_provider.py
--------------------
GPU / execution-provider detection for ONNX Runtime.
Priority: CUDA/ROCm (NVIDIA/AMD) → DirectML (Windows AMD/Intel) → CPU

Works on:
  - NVIDIA GPU  : CUDA via PyTorch + onnxruntime (standard build)
  - AMD GPU     : ROCm via PyTorch on Linux  (torch built with ROCm)
                  DirectML on Windows        (onnxruntime-directml)
  - Intel GPU   : DirectML on Windows        (onnxruntime-directml)
  - No GPU      : CPU fallback — always works, no crash

If you have an AMD or Intel GPU on Windows, run setup_gpu.py once:
    python setup_gpu.py
"""
import sys


# ── Device detection ──────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Returns 'cuda', 'directml', or 'cpu'.
    'cuda' covers both NVIDIA CUDA and AMD ROCm (both use the CUDA API in PyTorch).
    Never raises — always falls back to 'cpu' on any error.
    """
    # 1. CUDA (NVIDIA) or ROCm (AMD on Linux) — both surface as torch.cuda
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            backend = "ROCm/AMD" if _is_rocm(torch) else "CUDA/NVIDIA"
            print(f"[GPU] {backend} GPU detected: {name}")
            return "cuda"
    except Exception:
        pass

    # 2. DirectML (AMD / Intel on Windows) — needs onnxruntime-directml
    if sys.platform == "win32":
        try:
            import onnxruntime as ort
            if "DmlExecutionProvider" in ort.get_available_providers():
                print("[GPU] DirectML ready — AMD/Intel GPU will be used")
                return "directml"
            print(
                "[GPU] AMD/Intel GPU found but DirectML is NOT installed.\n"
                "      Run once from the project root, then restart:\n"
                "          python setup_gpu.py"
            )
        except Exception:
            pass

    print("[GPU] No GPU acceleration found — running on CPU")
    return "cpu"


def _is_rocm(torch) -> bool:
    """Return True if the running PyTorch was built with ROCm support."""
    try:
        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


# ── Provider list ─────────────────────────────────────────────────────────────

def get_ort_providers(device: str) -> list:
    """
    Return ordered ORT provider list for the given device.
    CPUExecutionProvider is always last so ORT silently falls back to CPU
    if the GPU provider fails to initialise instead of crashing.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]

    if device == "cuda":
        # ROCm builds of onnxruntime use ROCMExecutionProvider;
        # NVIDIA builds use CUDAExecutionProvider.
        # Try both so the same code works for both GPU brands.
        for provider in ("CUDAExecutionProvider", "ROCMExecutionProvider"):
            if provider in available:
                return [provider, "CPUExecutionProvider"]

    if device == "directml" and "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


# ── Session options ───────────────────────────────────────────────────────────

def make_session_options(num_threads: int = 1):
    """
    Conservative thread count per session so 3 parallel ONNX sessions
    don't starve each other on CPU. Ignored for GPU sessions.
    """
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = num_threads
        opts.intra_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return opts
    except Exception:
        return None


# ── Session configuration ─────────────────────────────────────────────────────

def configure_onnx_session(model, model_path: str, providers: list) -> bool:
    """
    Swap the ORT session inside a loaded Ultralytics YOLO model for one
    that uses the correct execution providers (GPU → CPU fallback).

    For ONNX models, Ultralytics ignores the device= argument passed to
    .predict(); the session providers set HERE control where inference runs.
    Never raises — returns False on failure so callers can log and continue.
    """
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()

        # Keep only providers that are actually in this ORT build
        filtered = [p for p in providers if p in available]
        if "CPUExecutionProvider" not in filtered:
            filtered.append("CPUExecutionProvider")

        opts        = make_session_options(num_threads=1)
        new_session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=filtered,
        )

        active = new_session.get_providers()[0] if new_session.get_providers() else "CPUExecutionProvider"

        # Primary path: model.model.session  (Ultralytics AutoBackend)
        backend = getattr(model, "model", None)
        if backend is not None and hasattr(backend, "session"):
            backend.session = new_session
            print(f"[GPU] Session ready → {active}")
            return True

        # Fallback paths for older Ultralytics versions
        for attr_path in ("session", "predictor.model.session"):
            obj = model
            for part in attr_path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "run"):
                parent = model
                parts  = attr_path.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_session)
                print(f"[GPU] Session ready → {active}")
                return True

        print("[GPU] Could not locate ONNX session — model will use default provider")
        return False

    except Exception as e:
        print(f"[GPU] Session setup failed ({e}) — falling back to CPU")
        return False