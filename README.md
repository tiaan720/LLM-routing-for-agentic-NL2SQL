# LLM-routing-for-agentic-NL2SQL
LLM routing ochestration framework used for the academic research on an NL2SQL agentic workflow. This framework studies on how LLMs can get router for the optimal performance for the mentioned agentic workflow. This repository includes the code and scientif studies applied to evalute the LLM routeres in the framework.


## Architecture Overview

![System Architecture](ds-masters-Full%20arch.png)

The system consists of three main routing approaches:

**1. Matrix Router (RC.1):** Trains a neural network on chatbot arena data to learn query-model performance patterns. Uses embedding similarity and gradient descent optimization to predict the best model for each query type.

**2. RAG Router (RC.2):** Leverages retrieval-augmented generation with a vector store of historical query-model performance data. Retrieves similar past queries and selects models based on winning patterns.

**3. Supervisor Agent Router (RC.3):** Uses a meta-LLM agent that analyzes query characteristics and selects appropriate models based on learned model descriptions and capabilities.

### Component Breakdown

- **Agent Component (AC.1):** Core SQL agent that generates and executes database queries using selected LLMs
- **LLM Component (LC.1):** Repository of available models (Vertex AI, Anthropic, OpenAI, Mistral, etc.)
- **Database Component (DC.1-DC.2):** PostgreSQL/BigQuery integration with schema discovery and query execution
- **Evaluation Component (EO.1-EO.2):** MLflow-based performance tracking with custom SQL accuracy metrics
- **Data Component (DC.3):** Chatbot arena dataset processing and vector store creation

### Process Flow

1. **User Input (UI.1-UI.2):** Query submission and agent configuration
2. **Router Selection:** System routes query through chosen routing strategy
3. **Model Selection:** Router determines optimal LLM based on query characteristics
4. **SQL Generation:** Selected model generates SQL queries via ReAct agent pattern
5. **Execution & Evaluation:** Query execution with performance metric collection
6. **System Output (SO.1):** Final results with cost and performance tracking

### MLFlow setup

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --no-serve-artifacts
```

**Note:** If you get a 403 Forbidden error, make sure to:
1. Kill any existing MLflow processes: `pkill -f "mlflow server"`
2. Use `127.0.0.1` instead of `0.0.0.0` as the host

Open this in the web to view mlflow:  `http://127.0.0.1:5000/`

### MLflow Data Backup (GCP Storage)

**Push MLflow data to GCP bucket:**
```bash
uv run python src/utils/mlflow-results-storage.py --push
```
This creates a timestamped backup of `mlflow.db` and `mlruns/` folder in the `mlflow-llm-routing` GCP bucket.

**Pull latest MLflow data from GCP bucket:**
```bash
uv run python src/utils/mlflow-results-storage.py --pull
```
This downloads the most recent backup and automatically cleans Windows artifact paths.

**Pull specific backup by date:**
```bash
uv run python src/utils/mlflow-results-storage.py --pull --date 2024-11-08_14-30-00
```

**Note:** Requires GCP authentication (see GCP setup section below).

Enabled ollama to work:

```bash
ollama serve
```
### Code setup

Get all imports to work:
```bash
uv run python -m pip install -e .
```

For CUDA PyTorch support (optional):  

Install CUDA PyTorch (requires NVIDIA GPU with CUDA 12.1+)
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

GCP setup:
```bash
gcloud auth application-default login
```

```bash
gcloud auth login
```

```bash
gcloud config set project research-su-llm-routing
```

### Windows local setup
Setup for local running on windows

```bash
.\bin\windows_local_setup.ps1
```
run this if there is trouble setting up the database data load:

```bash
.\bin\windows_load_postgres_data.ps1
```

# Running the MCP server
To run the MCP server, use the following command:

```bash
uv run python mcp/server.py
```
the HTTP MCP endpoint is at: `http://localhost:8000/mcp`

# Some research findings and readups that was help full

`https://lmsys.org/blog/2024-04-19-arena-hard/#full-leaderboard-with-gpt-4-turbo-as-judge`

## Arena-Hard Benchmark Results

The file `data/arena_hard_results.csv` contains a snapshot of selected Arena-Hard leaderboard scores (captured locally for analysis). These results were collected via the Arena-Hard automated pipeline as described by the LMSYS team. Refer to the upstream leaderboard for the most current standings.

Citation for the Arena-Hard pipeline:

```bibtex
@misc{arenahard2024,
    title  = {From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline},
    url    = {https://lmsys.org/blog/2024-04-19-arena-hard/},
    author = {Tianle Li* and Wei-Lin Chiang* and Evan Frick and Lisa Dunlap and Banghua Zhu and Joseph E. Gonzalez and Ion Stoica},
    month  = {April},
    year   = {2024}
}
```
