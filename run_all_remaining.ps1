$ErrorActionPreference = 'Stop'
Set-Location 'D:\fuyou1'
$py = 'C:\Espressif\tools\python\v5.2.6\venv\Scripts\python.exe'
$logs = 'D:\fuyou1\poc\output\rerun_logs_20260802'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$keyLine = Select-String -Path 'D:\fuyou1\sdkconfig.user' -Pattern 'CONFIG_MIMI_API_KEY' | Select-Object -First 1
if (-not $keyLine) { throw 'CONFIG_MIMI_API_KEY not found in sdkconfig.user' }
$env:DEEPSEEK_API_KEY = ($keyLine.Line -split '=',2)[1].Trim().Trim('"')

$steps = @(
  @{ label='synthetic_seed123';   args=@('poc\experiment.py','--real','--seed','123','--days','30','--output-dir','D:\fuyou1\poc\output\run_seed123') },
  @{ label='synthetic_seed999';   args=@('poc\experiment.py','--real','--seed','999','--days','30','--output-dir','D:\fuyou1\poc\output\run_seed999') },
  @{ label='strands_aruba1';      args=@('poc\experiment_strands.py','--seed','42','--days','30','--output-dir','D:\fuyou1\poc\output\strands_seed42') },
  @{ label='uci_v3_real_sensors'; args=@('poc\experiment_uci.py','--seed','42','--days','30','--output-dir','D:\fuyou1\poc\output\uci_v3_seed42') }
)

foreach ($s in $steps) {
  $out = Join-Path $logs ($s.label + '.log')
  $marker = Join-Path $logs ($s.label + '.marker')
  "[start] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') pid=$PID" | Set-Content -Path $marker -Encoding utf8
  & $py $s.args *> $out
  "[done] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') exit=$LASTEXITCODE" | Add-Content -Path $marker -Encoding utf8
}

"ALL_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Set-Content -Path (Join-Path $logs 'all_done.marker') -Encoding utf8
