#!/bin/bash

set -e



mkdir -p /home/vscode/.config/gcloud

# Try Unix-like path first (Mac/Linux)
if [ -d "${HOME}/.config/gcloud" ]; then
    echo "Found Unix-like gcloud config, creating bind mount..."
    sudo mount --bind "${HOME}/.config/gcloud" /home/vscode/.config/gcloud 2>/dev/null || echo "Bind mount failed, continuing..."
elif [ -d "/mnt/c/Users/${USER}/AppData/Roaming/gcloud" ]; then
    echo "Found Windows gcloud config via WSL, creating bind mount..."
    sudo mount --bind "/mnt/c/Users/${USER}/AppData/Roaming/gcloud" /home/vscode/.config/gcloud 2>/dev/null || echo "Bind mount failed, continuing..."
else
    echo "No host gcloud config found, will use container-only config"
fi

# Google Cloud CLI (https://cloud.google.com/sdk/docs/install#deb)
echo "Installing: gcloud"
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update
sudo apt-get install -y google-cloud-cli
sudo apt-get install -y google-cloud-cli-gke-gcloud-auth-plugin
echo "Finished install: gcloud"
echo "-----------------------------------"

# Install Chromium for Plotly PDF generation
echo "Installing: Chromium (required for Plotly PDF export)"
sudo apt-get update
sudo apt-get install -y chromium
echo "Finished install: Chromium"
echo "-----------------------------------"



echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH immediately
export PATH="/home/vscode/.local/bin:$PATH"

echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

echo "Creating new virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Verifying uv installation..."
which uv
uv --version

echo "Installing project dependencies..."
uv pip install --upgrade pip
uv pip compile pyproject.toml --extra dev --extra test -o requirements.txt
uv pip sync requirements.txt

# Verify dev tools installation
echo "Verifying dev tools..."
if ! .venv/bin/isort --version; then
    echo "isort installation failed"
    exit 1
fi
if ! .venv/bin/black --version; then
    echo "black installation failed"
    exit 1
fi


echo "Installing ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start ollama service in the background
echo "Starting ollama service..."
ollama serve > /dev/null 2>&1 &

# Wait for ollama service to start
echo "Waiting for ollama service to start..."
sleep 5

echo "Ollama setup complete!"

ollama pull snowflake-arctic-embed:22m

# Load PostgreSQL data
echo "Loading PostgreSQL data..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$WORKSPACE_DIR"
if ! bash "$SCRIPT_DIR/load_postgres_data.sh"; then
    echo "Fallback: trying absolute path for load_postgres_data.sh..."
    bash "/workspaces/DS-masters-2025-llm-routing/bin/load_postgres_data.sh"
fi

echo "Setup complete!"


