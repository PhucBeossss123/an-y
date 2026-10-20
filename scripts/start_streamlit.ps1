$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$url = 'http://localhost:8502'
$healthUrl = "$url/_stcore/health"
$logDir = Join-Path $repoRoot 'outputs\streamlit'
$stdoutLog = Join-Path $logDir 'stdout.log'
$stderrLog = Join-Path $logDir 'stderr.log'

trap {
    try {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        Add-Content -LiteralPath $stderrLog -Value ("[{0}] {1}" -f (Get-Date -Format s), $_.ToString()) -Encoding utf8
    } catch {
        # Keep the original launcher error if the log itself cannot be written.
    }
    exit 1
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Khong tim thay Python trong moi truong .venv: $python. Hay tao lai .venv va cai requirements-app.txt."
}

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:STREAMLIT_CONFIG_DIR = Join-Path $repoRoot '.streamlit'

try {
    Invoke-WebRequest -Uri $healthUrl -TimeoutSec 2 -UseBasicParsing | Out-Null
    Start-Process $url
    exit 0
} catch {
    # Chua co Streamlit san sang; bat tien trinh nen ben duoi.
}

$portOwner = Get-NetTCPConnection -LocalPort 8502 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($portOwner) {
    throw "Cong 8502 da duoc tien trinh PID $($portOwner.OwningProcess) su dung nhung Streamlit chua phan hoi. Xem $stderrLog."
}

$arguments = @(
    '-m', 'streamlit', 'run', 'app.py',
    '--server.address', '0.0.0.0',
    '--server.port', '8502',
    '--server.headless', 'true'
)
Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot `
    -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog | Out-Null

for ($attempt = 0; $attempt -lt 45; $attempt++) {
    Start-Sleep -Seconds 2
    try {
        Invoke-WebRequest -Uri $healthUrl -TimeoutSec 2 -UseBasicParsing | Out-Null
        Start-Process $url
        exit 0
    } catch {
        # Streamlit can take a few seconds to import dependencies and load reports.
    }
}

throw "Streamlit khong san sang sau 90 giay. Xem log: $stderrLog"
