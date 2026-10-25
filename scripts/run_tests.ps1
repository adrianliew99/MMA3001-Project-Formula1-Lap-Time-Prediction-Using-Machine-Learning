param(
    [string]$Python = "$PSScriptRoot/../.venv-tests/Scripts/python.exe"
)
$ErrorActionPreference = 'Stop'
Push-Location "$PSScriptRoot/.."
try {
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Python not found at $Python. Create .venv-tests and install requirements.txt and requirements-test.txt first."
    }
    New-Item -ItemType Directory -Force results/tests | Out-Null
    & $Python -m pytest --html=results/tests/report.html --self-contained-html --junitxml=results/tests/junit.xml --cov=mma3001_f1 --cov-report=term-missing --cov-report=html:results/tests/coverage --cov-report=xml:results/tests/coverage.xml @args
    $testExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $testExitCode
