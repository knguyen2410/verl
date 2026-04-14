import os
import subprocess
import sys

print("=== Environment ===")
print("Python executable:", sys.executable)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("NVIDIA_VISIBLE_DEVICES:", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("CUDA_PATH:", os.environ.get("CUDA_PATH"))

print("\n=== nvidia-smi ===")
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
         "--format=csv,noheader"],
        text=True
    )
    print(out.strip() if out.strip() else "No GPUs listed by nvidia-smi")
except Exception as e:
    print("nvidia-smi failed:", e)

print("\n=== PyTorch CUDA Check ===")
try:
    import torch
    print("torch version:", torch.__version__)
    print("torch cuda build:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())

    if torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:", torch.cuda.get_device_name(i))

    if torch.cuda.is_available():
        x = torch.randn(1024, device="cuda")
        print("CUDA tensor allocation: OK")
        print("Sample tensor device:", x.device)
    else:
        print("CUDA tensor allocation: SKIPPED (CUDA not available)")
except Exception as e:
    print("PyTorch check failed:", e)