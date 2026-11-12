$Project = "C:\Projects\TRACE_X\TRACE-X"
$LogFile = "$Project\results\trace_x_terminal_history.txt"

Set-Location $Project

Clear-Host

Write-Host "============================================"
Write-Host "       TRACE-X TERMINAL SESSION"
Write-Host "============================================"

# Display previous terminal output
if (Test-Path $LogFile) {
    Write-Host "`nPREVIOUS SESSION HISTORY`n"
    Get-Content $LogFile
}

Write-Host "`n============================================"
Write-Host "NEW TRACE-X SESSION"
Write-Host "============================================"

# Save all new terminal output
Start-Transcript -Path $LogFile -Append

Write-Host "Project: $Project"
Write-Host "Starting TRACE-X..."
Write-Host ""

python app.py

Stop-Transcript