# 1. Ensure D: drive cache directories exist
$baseCache = "D:\Lora++\.cache"
$tempDir = "$baseCache\temp"
$hfHome = "$baseCache\huggingface"
$torchHome = "$baseCache\torch"
$pipCache = "$baseCache\pip"
$pyCache = "$baseCache\pycache"

New-Item -ItemType Directory -Force -Path $tempDir, $hfHome, $torchHome, $pipCache, $pyCache | Out-Null

# 2. Redirect all temp and cache environment variables to D: drive
$env:TEMP = $tempDir
$env:TMP = $tempDir
$env:TMPDIR = $tempDir
$env:HF_HOME = $hfHome
$env:TRANSFORMERS_CACHE = $hfHome
$env:HF_DATASETS_CACHE = "$hfHome\datasets"
$env:TORCH_HOME = $torchHome
$env:PIP_CACHE_DIR = $pipCache
$env:PYTHONPYCACHEPREFIX = $pyCache

# 3. CPU/Memory safety limits to protect Windows OS memory limits
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:TOKENIZERS_PARALLELISM = "false"

# 4. Execute command
if ($args.Count -eq 0) {
    Write-Host "[+] D: Drive Cache and Memory Redirection Active!" -ForegroundColor Green
    Write-Host "    TEMP: $env:TEMP"
    Write-Host "    HF_HOME: $env:HF_HOME"
    Write-Host "    TORCH_HOME: $env:TORCH_HOME"
} else {
    & $args[0] $args[1..($args.Count - 1)]
}
