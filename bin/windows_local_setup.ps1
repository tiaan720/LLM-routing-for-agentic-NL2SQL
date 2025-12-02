# Windows Local Setup Script (PowerShell)
# Save as windows_local_setup.ps1 and run in PowerShell

Write-Host "Starting PostgreSQL with Docker Compose..."
docker-compose up -d postgres

Write-Host "Waiting for PostgreSQL to be ready..."
# Wait for PostgreSQL to be ready
docker-compose exec postgres bash -c "until pg_isready -U postgres; do sleep 1; done"

# Detect python executable
$python = "python"
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not installed or not in PATH. Please install Python 3.12+ and ensure it's in your PATH."
    exit 1
}

Write-Host "Setting up Python virtual environment..."
if (Test-Path ".venv") {
    Write-Host "Removing existing virtual environment..."
    Remove-Item -Recurse -Force .venv
}
python -m venv .venv

$activateScript = ".venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..."
    & $activateScript
} else {
    Write-Host "Could not find virtual environment activation script."
    exit 1
}

# Install uv if not present
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    python -m pip install --upgrade pip
    python -m pip install uv
}

Write-Host "Verifying uv installation..."
uv --version

Write-Host "Installing project dependencies..."
uv pip install --upgrade pip
uv pip compile pyproject.toml --extra dev --extra test -o requirements.txt
uv pip sync requirements.txt

# Verify dev tools installation
if (-not (Get-Command isort -ErrorAction SilentlyContinue)) {
    Write-Host "isort installation failed"
    exit 1
}
if (-not (Get-Command black -ErrorAction SilentlyContinue)) {
    Write-Host "black installation failed"
    exit 1
}

# Install and configure Ollama
Write-Host "Installing ollama..."
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Please install Ollama manually from https://ollama.com/download for Windows."
} else {
    Write-Host "Starting ollama service..."
    Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden
    Start-Sleep -Seconds 5
    
    # Check if Ollama is running with CUDA
    Write-Host "Checking Ollama GPU support..."
    try {
        $ollamaVersion = ollama --version 2>&1
        Write-Host "Ollama version: $ollamaVersion"
        
        # Check CUDA availability through nvidia-smi
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            Write-Host "NVIDIA GPU detected. Checking CUDA status..."
            $cudaInfo = nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "CUDA GPU Info: $cudaInfo"
                
                # Check GPU utilization and memory
                $gpuUtil = nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader 2>&1
                Write-Host "GPU Utilization: $gpuUtil"
                
                # Check if CUDA is available in system
                $cudaPath = $env:CUDA_PATH
                if ($cudaPath) {
                    Write-Host "CUDA installation found at: $cudaPath" -ForegroundColor Green
                }
                
                Write-Host "Ollama should be using CUDA acceleration." -ForegroundColor Green
            }
        } else {
            Write-Host "No NVIDIA GPU detected. Ollama will run on CPU." -ForegroundColor Yellow
        }
        
        # Check Ollama's runtime info
        $ollamaPs = ollama ps 2>&1
        if ($ollamaPs) {
            Write-Host "Ollama running models: $ollamaPs"
        }
        
        # Pull the model first
        Write-Host "Pulling snowflake-arctic-embed:22m model..."
        ollama pull snowflake-arctic-embed:22m
        
        # Test GPU acceleration with a small model
        Write-Host "Testing GPU acceleration..."
        $testOutput = ollama run snowflake-arctic-embed:22m "test" 2>&1
        
        # Check if the test ran successfully
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Model test completed successfully" -ForegroundColor Green
            
            # Monitor GPU during test
            $gpuDuringTest = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1
            if ($gpuDuringTest -and $LASTEXITCODE -eq 0) {
                Write-Host "GPU stats during test: $gpuDuringTest"
            }
        } else {
            Write-Host "Model test failed: $testOutput" -ForegroundColor Yellow
        }
        
        # Check Ollama logs for GPU usage
        $ollamaLogPath = "$env:USERPROFILE\.ollama\logs"
        if (Test-Path $ollamaLogPath) {
            $recentLog = Get-ChildItem $ollamaLogPath -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($recentLog) {
                $logContent = Get-Content $recentLog.FullName -Tail 50 | Select-String -Pattern "cuda|gpu|nvidia" -CaseSensitive:$false
                if ($logContent) {
                    Write-Host "GPU-related log entries found:" -ForegroundColor Green
                    $logContent | ForEach-Object { Write-Host $_ }
                }
            }
        }
        
        # Additional GPU confirmation
        $gpuProcesses = nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
        if ($LASTEXITCODE -eq 0 -and $gpuProcesses -match "ollama") {
            Write-Host "Ollama is actively using GPU!" -ForegroundColor Green
            Write-Host "GPU Process Info: $gpuProcesses"
        }
        
        # Final GPU status summary
        Write-Host "`n=== GPU Status Summary ===" -ForegroundColor Cyan
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            $gpuMemory = nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader 2>&1
            Write-Host "GPU Memory Available: $gpuMemory"
            Write-Host "Ollama is configured to use GPU acceleration when available." -ForegroundColor Green
        } else {
            Write-Host "Running in CPU mode." -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host "Could not determine Ollama GPU status: $_" -ForegroundColor Yellow
    }
}


# Load PostgreSQL data using the dedicated script
Write-Host "Loading PostgreSQL data..."
$postgresScript = Join-Path $PSScriptRoot "windows_load_postgres_data.ps1"
if (Test-Path $postgresScript) {
    & $postgresScript
} else {
    Write-Host "Warning: windows_load_postgres_data.ps1 not found at $postgresScript"
}

Write-Host "Setting up Git hooks..."
if (Test-Path "hooks") {
    # Copy all hooks to .git/hooks
    Copy-Item -Path "hooks\*" -Destination ".git\hooks\" -Force
    
    # Convert line endings to Unix format (LF) for Git Bash
    Get-ChildItem ".git\hooks\*" -File | ForEach-Object {
        $content = Get-Content $_.FullName -Raw
        $content = $content -replace "`r`n", "`n"
        [System.IO.File]::WriteAllText($_.FullName, $content)
    }
    
    Write-Host "Git hooks copied and configured successfully"
} else {
    Write-Host "Hooks directory not found, skipping git hooks setup"
}

Write-Host "Windows local setup complete!"