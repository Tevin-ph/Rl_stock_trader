param(
    [int]$epoch = 40
)

# Run from the script directory so relative paths in the project work
$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $here

Write-Host "Running evaluation for epoch $epoch..."
python .\evaluate.py --epoch $epoch

$plot = Join-Path $here "evaluation_plot.png"
if (Test-Path $plot) {
    Write-Host "Opening plot: $plot"
    Start-Process $plot
} else {
    Write-Host "Plot not found: $plot"
}

Pop-Location
