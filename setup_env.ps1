# Creates a dedicated conda env "molebot" (Python 3.13) and installs the RL stack.
# Run:  powershell -ExecutionPolicy Bypass -File setup_env.ps1
# Needs `conda` on PATH (Anaconda / Miniconda prompt).

$ErrorActionPreference = "Continue"
$envName = "molebot"

function Log($m) { $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m; Write-Output $line }

Log "=== STEP 1/5: create conda env '$envName' (python 3.13) ==="
conda create -n $envName python=3.13 -y
$envPy = (conda run -n $envName python -c "import sys; print(sys.executable)").Trim()
if (-not (Test-Path $envPy)) { Log "FATAL: env python not found"; exit 1 }
Log "env python: $envPy"
& $envPy --version

Log "=== STEP 2/5: upgrade pip ==="
& $envPy -m pip install --upgrade pip

Log "=== STEP 3/5: install PyTorch (CUDA 12.8 build for RTX 50xx / Blackwell) ==="
& $envPy -m pip install torch --index-url https://download.pytorch.org/whl/cu128

Log "=== STEP 4/5: install RL stack (SB3 + gymnasium + coppeliasim client + tools) ==="
& $envPy -m pip install -r requirements.txt cbor2

Log "=== STEP 5/5: verify imports ==="
& $envPy -c "import torch, stable_baselines3 as sb3, gymnasium, coppeliasim_zmqremoteapi_client as z; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('sb3', sb3.__version__); print('gymnasium', gymnasium.__version__)"

Log "=== SETUP DONE ==="
