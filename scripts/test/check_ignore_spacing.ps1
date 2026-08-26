<#
.SYNOPSIS
    Regulate intra-comment typing ignore spacing in Python files.
#>
param(
    [Parameter(Position=0, ValueFromRemainingArguments=$true)]
    [string[]]$Files,
    [switch]$Check,
    [switch]$Fix
)

if (-not $Files -or $Files.Length -eq 0) {
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $Files = git ls-files "*.py" "*.pyi"
    } else {
        $Files = Get-ChildItem -Recurse -Include *.py, *.pyi | ForEach-Object { $_.FullName }
    }
}

$Violations = 0
$ModifiedFiles = 0

foreach ($file in $Files) {
    if (-not (Test-Path $file -PathType Leaf)) { continue }

    $content = Get-Content -LiteralPath $file -Raw -Encoding utf8
    $lines = $content -split "`r?`n"
    $newLines = @()
    $fileChanged = $false

    foreach ($line in $lines) {
        $firstHash = $line.IndexOf("#")
        if ($firstHash -ge 0) {
            $codePart = $line.Substring(0, $firstHash)
            $commentPart = $line.Substring($firstHash)

            $newComment = [regex]::Replace($commentPart, "\]\s*#(?!#)", "] #")

            if ($newComment -ne $commentPart) {
                $fileChanged = $true
            }
            $newLines += ($codePart + $newComment)
        } else {
            $newLines += $line
        }
    }

    if ($fileChanged) {
        if ($Check) {
            Write-Host "Spacing violation in: $file"
            $Violations++
        } else {
            $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
            $ending = if ($content -match "`r`n") { "`r`n" } else { "`n" }
            $newContent = ($newLines -join $ending)
            if ($content.EndsWith("`n") -and -not $newContent.EndsWith($ending)) {
                $newContent += $ending
            }
            [System.IO.File]::WriteAllText((Convert-Path $file), $newContent, $utf8NoBom)
            Write-Host "Formatted: $file"
            $ModifiedFiles++
        }
    }
}

if ($Check -and $Violations -gt 0) {
    Write-Host "Found $Violations file(s) with comment spacing violations."
    exit 1
} elseif (-not $Check -and $ModifiedFiles -gt 0) {
    Write-Host "Reformatted $ModifiedFiles file(s)."
    exit 0
}

exit 0
