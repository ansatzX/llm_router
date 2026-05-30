# setup.ps1 - LLM Router Windows Setup
# Run: powershell -ExecutionPolicy Bypass -File scripts/setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== LLM Router Windows Setup ===" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python >= 3.10
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
try {
    $pyver = python --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "not found" }
    Write-Host "  Found: $pyver"
} catch {
    Write-Error "Python 3.10+ not found. Install from https://www.python.org/downloads/"
    exit 1
}

# 2. Check/Install uv
Write-Host "[2/6] Checking uv..." -ForegroundColor Yellow
$uvExists = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvExists) {
    Write-Host "  Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    Write-Host "  uv installed."
} else {
    $uvver = uv --version 2>&1
    Write-Host "  Found: $uvver"
}

# 3. Install dependencies
Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow
uv sync --extra windows
Write-Host "  Done."

# 4. Setup .env
Write-Host "[4/6] Checking .env..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "  Created .env from .env.example"
        Write-Host "  IMPORTANT: Edit .env with your API keys before running." -ForegroundColor Red
    } else {
        Write-Host "  No .env.example found, skipping."
    }
} else {
    Write-Host "  .env already exists."
}

# 5. Check router.toml
Write-Host "[5/6] Checking router.toml..." -ForegroundColor Yellow
if (-not (Test-Path "router.toml")) {
    Write-Host "  WARNING: No router.toml found. The router needs one to start." -ForegroundColor Red
} else {
    Write-Host "  router.toml found."
}

# 6. Optional desktop shortcut
Write-Host "[6/6] Desktop shortcut..." -ForegroundColor Yellow
$answer = Read-Host "  Create desktop shortcut? (y/n)"
if ($answer -eq "y") {
    $WshShell = New-Object -ComObject WScript.Shell
    $desktop = [System.Environment]::GetFolderPath("Desktop")
    $shortcut = $WshShell.CreateShortcut("$desktop\LLM Router.lnk")
    $shortcut.TargetPath = "uv"
    $shortcut.Arguments = "run llm-router gui"
    $shortcut.WorkingDirectory = (Get-Location).Path
    $shortcut.Description = "LLM Router - System Tray Application"
    $shortcut.Save()
    Write-Host "  Shortcut created on Desktop."
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Run the tray app:  uv run llm-router gui" -ForegroundColor Cyan
Write-Host "Run CLI only:      uv run llm-router serve" -ForegroundColor Cyan
Write-Host "Check status:      uv run llm-router status" -ForegroundColor Cyan
Write-Host ""
