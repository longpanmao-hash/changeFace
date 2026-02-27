param(
  [Parameter(Mandatory = $true, Position = 0)][string]$Source,
  [Parameter(Mandatory = $true, Position = 1)][string]$Target,
  [Parameter(Mandatory = $true, Position = 2)][string]$Output,
  [Parameter(Mandatory = $false, Position = 3)][ValidateSet("standard","fullface","headpaste","headreplace","features")][string]$Mode = "standard",
  [Parameter(Mandatory = $false, Position = 4)][string]$SourceRefs = "",
  [Parameter(Mandatory = $false, Position = 5)][string]$HeadStrength = "0.65"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Source)) { throw "Source not found: $Source" }
if (!(Test-Path $Target)) { throw "Target not found: $Target" }
if (!($Source.ToLower().EndsWith(".png"))) { throw "Source must be .png: $Source" }
if (!($Target.ToLower().EndsWith(".png"))) { throw "Target must be .png: $Target" }
if (!($Output.ToLower().EndsWith(".png"))) { throw "Output must be .png: $Output" }

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

if ($SourceRefs -match '^\d+(\.\d+)?$' -and $HeadStrength -eq "0.65") {
  $HeadStrength = $SourceRefs
  $SourceRefs = ""
}

if ([string]::IsNullOrWhiteSpace($SourceRefs)) {
  & $pythonPath deep/swap_insightface.py --source "$Source" --target "$Target" --output "$Output" --mode "$Mode" --head-strength "$HeadStrength"
}
else {
  & $pythonPath deep/swap_insightface.py --source "$Source" --target "$Target" --output "$Output" --mode "$Mode" --source-refs "$SourceRefs" --head-strength "$HeadStrength"
}
if ($LASTEXITCODE -ne 0) { throw "Deep swap failed." }
