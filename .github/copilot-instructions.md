# LLM Routing System

Data Science Masters project implementing intelligent LLM routing for SQL query generation using Matrix Router, RAG Router, and Supervisor Agent approaches. The system automatically selects optimal language models for SQL query tasks and tracks performance using MLflow.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

Bootstrap, build, and test the repository:

1. **Install uv package manager**: `pip install uv` -- takes ~10 seconds
2. **Create virtual environment**: `uv venv .venv` -- takes ~2 seconds  
3. **Install dependencies**: `uv sync --extra dev --extra test` -- takes ~60 seconds. NEVER CANCEL.
4. **Run tests**: `uv run pytest tests/ -v` -- takes ~15 seconds. NEVER CANCEL.
5. **Check code formatting**: 
   - `uv run black --check --diff src/ tests/` -- takes ~1 second
   - `uv run isort --check-only --diff src/ tests/` -- takes ~1 second

Start development services:

- **MLflow server**: `uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --no-serve-artifacts` -- starts in ~3 seconds. NEVER CANCEL. Set timeout to 60+ seconds.
- **MLflow web interface**: Available at `http://127.0.0.1:5000/`

## Validation

ALWAYS run through at least one complete end-to-end scenario after making changes:

1. **Test imports**: `uv run python -c "import src; print('Successfully imported src module')"`
2. **Test model registry**: `uv run pytest tests/test_new_model_system.py::TestModelRegistry::test_list_providers -v`
3. **Run all tests**: `uv run pytest tests/ -v` -- runs 11 tests, ~15 seconds
4. **Format code**: 
   - `uv run isort src/ tests/`
   - `uv run black src/ tests/`

ALWAYS run `uv run black src/ tests/` and `uv run isort src/ tests/` before committing changes or the pre-commit hooks will fail.

## Common Tasks

### Environment Setup
The project uses `uv` as the package manager with Python 3.11+. All dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

### Key Commands
```bash
# Full environment setup
uv sync --extra dev --extra test

# Run specific tests
uv run pytest tests/test_new_model_system.py -v

# Format code  
uv run black src/ tests/
uv run isort src/ tests/

# Start MLflow tracking server
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --no-serve-artifacts
```

### Project Structure
- **src/agents/**: SQL agents with different routing strategies
- **src/routers/**: Matrix router and RAG router implementations  
- **src/models/**: Model registry and provider abstractions
- **experiments/**: Evaluation scripts and Jupyter notebooks
- **tests/**: Test suite with model registry tests
- **mcp/**: MCP (Model Context Protocol) server implementation

### Development Services
- **PostgreSQL**: Database for SQL query experiments (requires Docker Compose)
- **MLflow**: Experiment tracking at http://127.0.0.1:5000/
- **Ollama**: Local model serving (not available in this environment)

### Important Notes
- The MCP server requires data preparation: `data/llm_examples/examples_vector_store.pkl` must be created before running
- PostgreSQL database setup requires Docker and data loading scripts from `bin/` directory
- Git hooks automatically format code with black and isort on commit
- The project supports multiple LLM providers: OpenAI, Vertex AI, Anthropic, Mistral, etc.

### Known Limitations
- Docker services (PostgreSQL, Ollama) may not be available in all environments
- Some integration tests require external API credentials
- Vector store files need to be generated before running certain components

## Frequently Used File Locations

### Repository Root
```
.
├── README.md                    # Project overview and setup instructions
├── pyproject.toml              # Python project configuration and dependencies
├── uv.lock                     # Locked dependency versions
├── requirements.txt            # Compiled requirements (auto-generated)
├── docker-compose.yml          # PostgreSQL service definition
├── .env_example               # Environment variables template
├── src/                       # Main source code
├── tests/                     # Test suite
├── experiments/               # Research experiments and evaluations
├── bin/                       # Setup and utility scripts
├── data/                      # Database schemas and example data
└── .github/                   # GitHub configuration
```

### Source Code Structure
```
src/
├── agents/                    # SQL generation agents
│   ├── sql_agent.py          # Base SQL agent implementation
│   ├── sql_agent_matrix_router.py  # Matrix routing strategy
│   └── sql_agent_rag_router.py     # RAG routing strategy
├── routers/                   # Routing algorithm implementations
│   ├── matrix_router.py      # Neural network-based routing
│   └── rag_router.py         # Retrieval-augmented routing
├── models/                    # Model registry and providers
└── retrievers/               # Vector store and retrieval components
```

### Common Command Outputs

#### Test Output (uv run pytest tests/ -v)
```
11 tests collected
TestModelRegistry::test_list_providers PASSED
TestModelRegistry::test_is_registered PASSED  
TestModelRegistry::test_registry_consistency PASSED
TestModelCreation::test_create_vertex_ai_model PASSED
[...additional tests...]
10 passed, 1 skipped, 5 warnings in 10.03s
```

#### Available Providers (from model registry)
- openai
- vertex_ai  
- vertex_anthropic
- vertex_model_garden
- mistral
- ollama
- litellm

#### MLflow Server Startup
```
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
[2025-08-09 17:13:18 +0000] [3795] [INFO] Starting gunicorn 23.0.0
[2025-08-09 17:13:18 +0000] [3795] [INFO] Listening at: http://127.0.0.1:5000 (3795)
```