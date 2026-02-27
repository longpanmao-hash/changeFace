param()

$ErrorActionPreference = "Stop"

function Resolve-Python {
  $candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
  )
  foreach ($p in $candidates) {
    if (Test-Path $p) { return $p }
  }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source -notlike "*WindowsApps\\python.exe") { return $cmd.Source }
  throw "No usable Python found. Install Python 3.10/3.11/3.12."
}

$pythonPath = Resolve-Python

& $pythonPath -c "import sys; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw "Python is not working correctly." }

$pyVersion = & $pythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) { throw "Failed to detect Python version." }
if ($pyVersion -notin @("3.10", "3.11", "3.12")) {
  throw "Unsupported Python version: $pyVersion. Please use Python 3.10/3.11/3.12 for InsightFace."
}

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (!(Test-Path $vcvars)) {
  throw "Microsoft C++ Build Tools not found. Install and include VCTools workload."
}

cmd /c "`"$vcvars`" >nul && `"$pythonPath`" -m pip install -r deep/requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "pip install failed." }

& $pythonPath deep/download_model.py
if ($LASTEXITCODE -ne 0) { throw "model download failed." }

Write-Host "Deep model setup completed."
