"""Print locally available Python/ML runtime details without project imports."""

import platform

import torch
import transformers
import sklearn


print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"CUDA available to PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
