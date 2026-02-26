"""
core/gpu_provider.py
--------------------
GPU / execution-provider detection for ONNX Runtime.
Priority: CUDA (NVIDIA) → DirectML (Windows AMD/Intel) → CPU

CPU fallback is always available — if no GPU provider loads the app
runs normally on CPU, no crash, no manual intervention needed.

AMD / Intel GPU users (Windows):
  The standard onnxruntime package does NOT include DirectML.
  Run setup_gpu.py once from the project root, then restart:
      python setup_gpu.py
"""
import sys


# ── Device detection ──────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Returns 'cuda', 'directml', or 'cpu'.
    Never raises — always falls back to 'cpu' on any error.
    """
    # 1. CUDA (NVIDIA)
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[GPU] CUDA available: {name}")
            return "cuda"
    except Exception:
        pass

    # 2. DirectML (AMD / Intel — Windows only)
    if sys.platform == "win32":
        try:
            import onnxruntime as ort
            if "DmlExecutionProvider" in ort.get_available_providers():
                print("[GPU] DirectML ready — AMD/Intel GPU will be used")
                return "directml"
            # DML not present — hint the user but don't crash
            print(
                "[GPU] AMD/Intel GPU detected but DirectML is NOT available.\n"
                "      Run once from the project root, then restart:\n"
                "          python setup_gpu.py"
            )
        except Exception:
            pass

    print("[GPU] Running on CPU")
    return "cpu"


# ── Provider list ─────────────────────────────────────────────────────────────

def get_ort_providers(device: str) -> list:
    """
    Return ordered ORT provider list for the given device.
    CPUExecutionProvider is always appended as the final fallback so that
    even if the GPU provider fails to load at session-creation time, ORT
    silently drops to CPU instead of raising.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]

    if device == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "directml" and "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    # CPU fallback — always valid
    return ["CPUExecutionProvider"]


# ── Session options ───────────────────────────────────────────────────────────

def make_session_options(num_threads: int = 1):
    """
    Conservative thread count per session.
    1 thread/session keeps 3 parallel ONNX sessions from starving the CPU.
    Ignored for GPU sessions (inference runs on-device).
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
    Swap the ORT session inside a loaded Ultralytics YOLO model for one that
    uses the correct execution providers (GPU → CPU fallback).

    - For ONNX models, Ultralytics' device= argument to .predict() is ignored;
      the session providers set HERE control where inference actually runs.
    - CPUExecutionProvider is always included so ORT can fall back silently if
      the GPU provider fails to initialise for any reason.
    - Never raises — returns False on failure so callers can log and continue.
    """
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()

        # Filter to what's actually in this ORT build; always keep CPU last
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

        print(f"[GPU] Could not locate ONNX session — model will use default provider")
        return False

    except Exception as e:
        print(f"[GPU] Session setup failed ({e}) — falling back to CPU")
        return False