$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$baseDir = Split-Path -Parent $scriptDir
$sourceDir = Join-Path $baseDir "cic-7%"
$stage1Dir = Join-Path $sourceDir "output_stage1"
$stage2Dir = Join-Path $sourceDir "output_stage2"
$stage3Dir = Join-Path $sourceDir "output_stage3"
$culture = [System.Globalization.CultureInfo]::InvariantCulture

function Get-ReportProperty {
    param(
        [object] $Report,
        [string] $Name
    )

    $property = $Report.PSObject.Properties[$Name]
    if (-not $property) {
        throw "Could not find property '$Name' in report."
    }
    return $property.Value
}

function Load-ClassificationReport {
    param(
        [string] $Path,
        [bool] $Nested
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing report: $Path"
    }

    $obj = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($Nested) {
        return Get-ReportProperty -Report $obj -Name "classification_report"
    }

    return $obj
}

function Load-ClassificationReportText {
    param([string] $Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing report: $Path"
    }

    $report = @{}
    $lines = Get-Content -LiteralPath $Path
    foreach ($line in $lines) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) {
            continue
        }

        if ($trimmed -match "^(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$") {
            $report[$matches[1]] = @{
                precision = [double]::Parse($matches[2], $culture)
                recall = [double]::Parse($matches[3], $culture)
                "f1-score" = [double]::Parse($matches[4], $culture)
                support = [int]::Parse($matches[5], $culture)
            }
        } elseif ($trimmed -match "^(macro avg|weighted avg)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$") {
            $report[$matches[1]] = @{
                precision = [double]::Parse($matches[2], $culture)
                recall = [double]::Parse($matches[3], $culture)
                "f1-score" = [double]::Parse($matches[4], $culture)
                support = [int]::Parse($matches[5], $culture)
            }
        } elseif ($trimmed -match "^accuracy\s+([0-9.]+)\s+(\d+)$") {
            $report["accuracy"] = [double]::Parse($matches[1], $culture)
        }
    }

    return ($report | ConvertTo-Json -Depth 5 | ConvertFrom-Json)
}

function Format-Value {
    param([double] $Value)
    return $Value.ToString("0.0000", $culture)
}

function Get-ClassF1 {
    param(
        [hashtable] $Reports,
        [string] $Model,
        [int] $Id
    )

    if (-not $Reports.ContainsKey($Model) -or $null -eq $Reports[$Model]) {
        return "N/A"
    }

    $classReport = Get-ReportProperty -Report $Reports[$Model] -Name ([string]$Id)
    return Format-Value ([double](Get-ReportProperty -Report $classReport -Name "f1-score"))
}

function Get-WeightedSummary {
    param(
        [hashtable] $Reports,
        [string] $Model,
        [string] $Metric
    )

    if (-not $Reports.ContainsKey($Model) -or $null -eq $Reports[$Model]) {
        return "N/A"
    }

    if ($Metric -eq "accuracy") {
        return Format-Value ([double](Get-ReportProperty -Report $Reports[$Model] -Name "accuracy"))
    }

    $weighted = Get-ReportProperty -Report $Reports[$Model] -Name "weighted avg"
    return Format-Value ([double](Get-ReportProperty -Report $weighted -Name $Metric))
}

function Escape-Latex {
    param([string] $Text)

    return $Text.Replace("\", "\textbackslash{}").
        Replace("&", "\&").
        Replace("%", "\%").
        Replace("$", "\$").
        Replace("#", "\#").
        Replace("_", "\_").
        Replace("{", "\{").
        Replace("}", "\}")
}

function Draw-Text {
    param(
        [System.Drawing.Graphics] $Graphics,
        [string] $Text,
        [System.Drawing.Font] $Font,
        [System.Drawing.Brush] $Brush,
        [System.Drawing.RectangleF] $Rect,
        [System.Drawing.StringAlignment] $Horizontal = [System.Drawing.StringAlignment]::Near,
        [System.Drawing.StringAlignment] $Vertical = [System.Drawing.StringAlignment]::Center
    )

    $format = [System.Drawing.StringFormat]::new()
    $format.Alignment = $Horizontal
    $format.LineAlignment = $Vertical
    $format.Trimming = [System.Drawing.StringTrimming]::EllipsisCharacter
    $Graphics.DrawString($Text, $Font, $Brush, $Rect, $format)
    $format.Dispose()
}

function New-Rect {
    param([float] $X, [float] $Y, [float] $W, [float] $H)
    return [System.Drawing.RectangleF]::new($X, $Y, $W, $H)
}

function Write-StageTable {
    param(
        [string] $StageTitle,
        [string] $Caption,
        [string] $OutputStem,
        [hashtable] $Reports,
        [array] $Groups
    )

    $texPath = Join-Path $scriptDir "$OutputStem.tex"
    $pngPath = Join-Path $scriptDir "$OutputStem.png"

    $texLines = New-Object System.Collections.Generic.List[string]
    $texLines.Add("\begin{table}[htbp]")
    $texLines.Add("\centering")
    $texLines.Add("\caption{$Caption}")
    $texLines.Add("\label{tab:$OutputStem}")
    $texLines.Add("\small")
    $texLines.Add("\setlength{\tabcolsep}{8pt}")
    $texLines.Add("\renewcommand{\arraystretch}{1.16}")
    $texLines.Add("\begin{tabular}{llcccc}")
    $texLines.Add("\toprule")
    $texLines.Add("\multicolumn{6}{c}{\textbf{$StageTitle}} \\")
    $texLines.Add("\midrule")
    $texLines.Add("\textbf{Attacks} & \textbf{Class label} & \multicolumn{4}{c}{\textbf{F1-score}} \\")
    $texLines.Add("\cmidrule(lr){3-6}")
    $texLines.Add(" & & \textbf{RF} & \textbf{XGB} & \textbf{Cat} & \textbf{LGB} \\")
    $texLines.Add("\midrule")

    foreach ($group in $Groups) {
        $count = $group.Items.Count
        for ($i = 0; $i -lt $count; $i++) {
            $item = $group.Items[$i]
            $attack = ""
            if ($i -eq 0) {
                $escapedAttack = Escape-Latex $group.Attack
                if ($count -gt 1) {
                    $attack = "\multirow{$count}{*}{\textbf{$escapedAttack}}"
                } else {
                    $attack = "\textbf{$escapedAttack}"
                }
            }

            $label = Escape-Latex $item.Label
            $id = [int]$item.Id
            $texLines.Add("$attack & $label & $(Get-ClassF1 $Reports RF $id) & $(Get-ClassF1 $Reports XGB $id) & $(Get-ClassF1 $Reports Cat $id) & $(Get-ClassF1 $Reports LGB $id) \\")
        }
        $texLines.Add("\midrule")
    }

    $texLines.Add("\multicolumn{2}{c}{\textbf{Accuracy}} & $(Get-WeightedSummary $Reports RF accuracy) & $(Get-WeightedSummary $Reports XGB accuracy) & $(Get-WeightedSummary $Reports Cat accuracy) & $(Get-WeightedSummary $Reports LGB accuracy) \\")
    $texLines.Add("\midrule")
    $texLines.Add("\multicolumn{2}{c}{\textbf{Weighted precision}} & $(Get-WeightedSummary $Reports RF precision) & $(Get-WeightedSummary $Reports XGB precision) & $(Get-WeightedSummary $Reports Cat precision) & $(Get-WeightedSummary $Reports LGB precision) \\")
    $texLines.Add("\midrule")
    $texLines.Add("\multicolumn{2}{c}{\textbf{Weighted recall}} & $(Get-WeightedSummary $Reports RF recall) & $(Get-WeightedSummary $Reports XGB recall) & $(Get-WeightedSummary $Reports Cat recall) & $(Get-WeightedSummary $Reports LGB recall) \\")
    $texLines.Add("\midrule")
    $texLines.Add("\multicolumn{2}{c}{\textbf{Weighted F1}} & $(Get-WeightedSummary $Reports RF "f1-score") & $(Get-WeightedSummary $Reports XGB "f1-score") & $(Get-WeightedSummary $Reports Cat "f1-score") & $(Get-WeightedSummary $Reports LGB "f1-score") \\")
    $texLines.Add("\bottomrule")
    $texLines.Add("\end{tabular}")
    $texLines.Add("\end{table}")

    Set-Content -LiteralPath $texPath -Value $texLines -Encoding UTF8

    Add-Type -AssemblyName System.Drawing

    $margin = 32
    $colWidths = @(210, 420, 165, 165, 165, 165)
    $tableWidth = ($colWidths | Measure-Object -Sum).Sum
    $width = [int]($tableWidth + 2 * $margin)
    $rowHeight = 42
    $titleHeight = 54
    $headerOneHeight = 58
    $headerTwoHeight = 44
    $dataRowCount = ($Groups | ForEach-Object { $_.Items.Count } | Measure-Object -Sum).Sum
    $summaryRowCount = 4
    $height = [int](2 * $margin + $titleHeight + $headerOneHeight + $headerTwoHeight + ($dataRowCount + $summaryRowCount) * $rowHeight + 8)

    $bitmap = [System.Drawing.Bitmap]::new($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $graphics.Clear([System.Drawing.Color]::White)

    $black = [System.Drawing.Brushes]::Black
    $linePen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(60, 60, 60), 1.4)
    $heavyPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(45, 45, 45), 2.0)
    $font = [System.Drawing.Font]::new("Times New Roman", 19, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
    $boldFont = [System.Drawing.Font]::new("Times New Roman", 19, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    $titleFont = [System.Drawing.Font]::new("Times New Roman", 23, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)

    $x = @($margin)
    for ($i = 1; $i -lt $colWidths.Count; $i++) {
        $x += ($x[$i - 1] + $colWidths[$i - 1])
    }

    $y = [float]$margin
    $graphics.DrawLine($heavyPen, $margin, $y, $margin + $tableWidth, $y)
    Draw-Text $graphics $StageTitle $titleFont $black (New-Rect $margin $y $tableWidth $titleHeight) ([System.Drawing.StringAlignment]::Center)
    $y += $titleHeight
    $graphics.DrawLine($linePen, $margin, $y, $margin + $tableWidth, $y)

    $headerTop = $y
    Draw-Text $graphics "Attacks" $boldFont $black (New-Rect $x[0] $headerTop $colWidths[0] ($headerOneHeight + $headerTwoHeight)) ([System.Drawing.StringAlignment]::Near)
    Draw-Text $graphics "Class label" $boldFont $black (New-Rect $x[1] $headerTop $colWidths[1] ($headerOneHeight + $headerTwoHeight)) ([System.Drawing.StringAlignment]::Near)
    Draw-Text $graphics "F1-score" $boldFont $black (New-Rect $x[2] $headerTop ($colWidths[2] + $colWidths[3] + $colWidths[4] + $colWidths[5]) $headerOneHeight) ([System.Drawing.StringAlignment]::Center)
    $graphics.DrawLine($linePen, $x[2], $headerTop + $headerOneHeight, $margin + $tableWidth, $headerTop + $headerOneHeight)

    $modelY = $headerTop + $headerOneHeight
    foreach ($modelIndex in 0..3) {
        $columnIndex = $modelIndex + 2
        $modelLabel = @("RF", "XGB", "Cat", "LGB")[$modelIndex]
        Draw-Text $graphics $modelLabel $boldFont $black (New-Rect $x[$columnIndex] $modelY $colWidths[$columnIndex] $headerTwoHeight) ([System.Drawing.StringAlignment]::Center)
    }

    $y += $headerOneHeight + $headerTwoHeight
    $graphics.DrawLine($linePen, $margin, $y, $margin + $tableWidth, $y)

    foreach ($group in $Groups) {
        $groupY = $y
        $spanHeight = $group.Items.Count * $rowHeight
        Draw-Text $graphics $group.Attack $boldFont $black (New-Rect ($x[0] + 12) $groupY ($colWidths[0] - 20) $spanHeight)

        foreach ($item in $group.Items) {
            Draw-Text $graphics $item.Label $font $black (New-Rect ($x[1] + 12) $y ($colWidths[1] - 20) $rowHeight)
            $id = [int]$item.Id
            $values = @((Get-ClassF1 $Reports RF $id), (Get-ClassF1 $Reports XGB $id), (Get-ClassF1 $Reports Cat $id), (Get-ClassF1 $Reports LGB $id))
            foreach ($modelIndex in 0..3) {
                $columnIndex = $modelIndex + 2
                Draw-Text $graphics $values[$modelIndex] $font $black (New-Rect $x[$columnIndex] $y $colWidths[$columnIndex] $rowHeight) ([System.Drawing.StringAlignment]::Center)
            }
            $y += $rowHeight
        }

        $graphics.DrawLine($linePen, $margin, $y, $margin + $tableWidth, $y)
    }

    $summaryRows = @(
        @{ Label = "Accuracy"; Values = @((Get-WeightedSummary $Reports RF accuracy), (Get-WeightedSummary $Reports XGB accuracy), (Get-WeightedSummary $Reports Cat accuracy), (Get-WeightedSummary $Reports LGB accuracy)) },
        @{ Label = "Weighted precision"; Values = @((Get-WeightedSummary $Reports RF precision), (Get-WeightedSummary $Reports XGB precision), (Get-WeightedSummary $Reports Cat precision), (Get-WeightedSummary $Reports LGB precision)) },
        @{ Label = "Weighted recall"; Values = @((Get-WeightedSummary $Reports RF recall), (Get-WeightedSummary $Reports XGB recall), (Get-WeightedSummary $Reports Cat recall), (Get-WeightedSummary $Reports LGB recall)) },
        @{ Label = "Weighted F1"; Values = @((Get-WeightedSummary $Reports RF "f1-score"), (Get-WeightedSummary $Reports XGB "f1-score"), (Get-WeightedSummary $Reports Cat "f1-score"), (Get-WeightedSummary $Reports LGB "f1-score")) }
    )

    foreach ($summary in $summaryRows) {
        Draw-Text $graphics $summary.Label $boldFont $black (New-Rect $margin $y ($colWidths[0] + $colWidths[1]) $rowHeight) ([System.Drawing.StringAlignment]::Center)
        foreach ($modelIndex in 0..3) {
            $columnIndex = $modelIndex + 2
            Draw-Text $graphics $summary.Values[$modelIndex] $font $black (New-Rect $x[$columnIndex] $y $colWidths[$columnIndex] $rowHeight) ([System.Drawing.StringAlignment]::Center)
        }
        $y += $rowHeight
        $graphics.DrawLine($linePen, $margin, $y, $margin + $tableWidth, $y)
    }

    $graphics.DrawLine($heavyPen, $margin, $y, $margin + $tableWidth, $y)
    $bitmap.Save($pngPath, [System.Drawing.Imaging.ImageFormat]::Png)

    $font.Dispose()
    $boldFont.Dispose()
    $titleFont.Dispose()
    $linePen.Dispose()
    $heavyPen.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()

    Write-Host "Wrote: $texPath"
    Write-Host "Wrote: $pngPath"
}

$groups = @(
    @{
        Attack = "Benign"
        Items = @(@{ Label = "Benign"; Id = 0 })
    },
    @{
        Attack = "Botnets"
        Items = @(@{ Label = "Bot"; Id = 1 })
    },
    @{
        Attack = "DDoS"
        Items = @(
            @{ Label = "HOIC"; Id = 6 },
            @{ Label = "LOIC-UDP"; Id = 7 },
            @{ Label = "LOIC-HTTP"; Id = 8 }
        )
    },
    @{
        Attack = "DoS"
        Items = @(
            @{ Label = "GoldenEye"; Id = 9 },
            @{ Label = "Hulk"; Id = 10 },
            @{ Label = "SlowHTTPTest"; Id = 11 },
            @{ Label = "Slowloris"; Id = 12 }
        )
    },
    @{
        Attack = "BruteForce"
        Items = @(
            @{ Label = "FTP-BruteForce"; Id = 13 },
            @{ Label = "SSH-Bruteforce"; Id = 17 }
        )
    },
    @{
        Attack = "Infiltration"
        Items = @(
            @{ Label = "Infilteration"; Id = 14 },
            @{ Label = "Infiltration"; Id = 15 }
        )
    },
    @{
        Attack = "Web Attack"
        Items = @(
            @{ Label = "Brute Force -Web"; Id = 2 },
            @{ Label = "Brute Force -XSS"; Id = 3 },
            @{ Label = "Brute Force-Web"; Id = 4 },
            @{ Label = "Brute Force-XSS"; Id = 5 },
            @{ Label = "SQL Injection"; Id = 16 },
            @{ Label = "Web attack-SQL Injection"; Id = 18 }
        )
    }
)

$stage1Reports = @{
    RF = Load-ClassificationReport -Path (Join-Path $stage1Dir "cic_7percentoutput_stage1_RandomForest_GPU\metrics_RandomForest_GPU.json") -Nested $true
    XGB = Load-ClassificationReport -Path (Join-Path $stage1Dir "cic_7percentoutput_stage1_XGBoost_GPU\metrics_XGBoost_GPU.json") -Nested $true
    Cat = Load-ClassificationReport -Path (Join-Path $stage1Dir "cic_7percentoutput_stage1_CatBoost_GPU\metrics_CatBoost_GPU.json") -Nested $true
    LGB = Load-ClassificationReport -Path (Join-Path $stage1Dir "cic_7percentoutput_stage1_LightGBM_CPU\metrics_LightGBM_CPU.json") -Nested $true
}

$stage2Reports = @{
    RF = Load-ClassificationReportText -Path (Join-Path $stage2Dir "RF\rf_report.txt")
    XGB = Load-ClassificationReport -Path (Join-Path $stage2Dir "XGBoost\xgboost_report.json") -Nested $false
    Cat = Load-ClassificationReport -Path (Join-Path $stage2Dir "CatBoost\catboost_report.json") -Nested $false
    LGB = Load-ClassificationReport -Path (Join-Path $stage2Dir "LightGBM\lightgbm_report.json") -Nested $false
}

$stage3Reports = @{
    RF = Load-ClassificationReport -Path (Join-Path $stage3Dir "rf_report.json") -Nested $false
    XGB = Load-ClassificationReport -Path (Join-Path $stage3Dir "xgb_report.json") -Nested $false
    Cat = Load-ClassificationReport -Path (Join-Path $stage3Dir "cat_report.json") -Nested $false
    LGB = Load-ClassificationReport -Path (Join-Path $stage3Dir "lgb_report.json") -Nested $false
}

Write-StageTable `
    -StageTitle "CIC-7% Stage I dataset" `
    -Caption "Stage I F1-score comparison on the CIC-7\% dataset. Summary precision, recall, and F1 use weighted averages." `
    -OutputStem "cic7_stage1_table" `
    -Reports $stage1Reports `
    -Groups $groups

Write-StageTable `
    -StageTitle "CIC-7% Stage II dataset" `
    -Caption "Stage II F1-score comparison after feature selection and resampling on the CIC-7\% dataset. Summary precision, recall, and F1 use weighted averages." `
    -OutputStem "cic7_stage2_table" `
    -Reports $stage2Reports `
    -Groups $groups

Write-StageTable `
    -StageTitle "CIC-7% Stage III dataset" `
    -Caption "Stage III F1-score comparison after model optimization on the CIC-7\% dataset. Summary precision, recall, and F1 use weighted averages." `
    -OutputStem "cic7_stage3_table" `
    -Reports $stage3Reports `
    -Groups $groups
