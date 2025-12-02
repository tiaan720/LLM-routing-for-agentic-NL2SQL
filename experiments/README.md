# Experiments

This folder contains the experimental framework for evaluating LLM routing strategies for SQL agent tasks. The experiments compare different routing approaches to determine the most cost-effective and accurate model selection strategy.

## Overview

The experimental workflow evaluates three routing strategies:
- **Matrix Router**: Uses a trained ML model to predict optimal LLM based on query characteristics
- **RAG Router**: Uses semantic similarity search against historical query-model performance data
- **Supervisor Agent**: Uses an LLM-based supervisor to analyze queries and select appropriate models

## Quick Start

### Prerequisites

Start MLflow tracking server:
```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --no-serve-artifacts
```

Start Ollama (if using local models):
```bash
ollama serve
```

### Running Experiments

Follow the `experiment_flow.ipynb` notebook which orchestrates the complete experimental pipeline. The notebook provides an interactive interface to run experiments in the correct sequence.

## Experimental Workflow

### 1. Dataset Generation (`experiment_dataset_gen.py`)

Generates baseline performance data by running all available LLMs against test queries:

```python
run_all_model_evaluations(selected_datasets=['academic', 'atis', 'broker'])
```

- Executes each model against comprehensive test sets
- Records accuracy metrics, execution time, and costs
- Logs results to MLflow with experiment name `dataset_gen_{model_name}`
- Supports filtering by specific models/datasets and adding validation prefixes

**Key Metrics Tracked:**
- SQL accuracy (execution correctness)
- Answer correctness (semantic similarity to expected answer)
- Faithfulness (answer grounding in context)
- Query cost (input/output token costs)
- Execution time

### 2. Arena Competition (`experiment_dataset_gen_arena.py`)

Creates competitive pairwise comparisons to identify top-performing models:

```python
main()
```

- Tag runs in MLflow with `arena:arena` to include in competition
- Performs head-to-head comparisons between models per query
- Outputs `model_arena_results.csv` and `model_arena_results_matrix_router.csv`
- Used as training data for routing strategies

### 3. Router Evaluation

#### Matrix Router (`matrix_router_agent_eval.py`)

Trains a gradient boosting classifier to predict optimal model:

```python
router = ModelRouter()
router.train(router.results_csv_path, test_size=0.2, epochs=5)
results = matrix_router_main(datasets=['atis', 'broker', 'yelp'])
```

- Embeds queries using sentence transformers
- Learns from historical query-model performance data
- Implementation: `src/routers/matrix_router.py`
- Agent integration: `src/agents/sql_agent_matrix_router.py`

#### RAG Router (`rag_router_agent_eval.py`)

Uses semantic search to find similar historical queries:

```python
load_data(csv_file="data/model_arena_results.csv", 
          save_path="data/chatbot_arena_inmemory_vectorstore.pkl",
          recreate=True)
results = rag_router_main(datasets=['atis', 'broker', 'yelp'])
```

- Embeds arena results into vector store
- Retrieves top-k similar queries and their winning models
- Implementation: `src/routers/rag_router.py`
- Agent integration: `src/agents/sql_agent_rag_router.py`

#### Supervisor Agent (`supervisor_agent_eval.py`)

Uses an LLM to analyze queries and select models:

```python
results = supervisor_main(datasets=['atis', 'broker', 'yelp'])
```

- Leverages model descriptions from `configs/llm_model_descriptions.json`
- Uses reasoning to match query requirements to model capabilities
- Implementation: `src/agents/supervisor_agent.py`
- Delegates to SQL agents via tool calling

### 4. Validation & Analysis

#### Model Performance Analysis (`model_analysis_plots.py`)

Visualizes performance metrics across experiments:

```python
analyze_models_by_tag('arena', 'arena', 'arena', '')
analyze_models_by_tag('val', 'val', 'validation', ' - Validation Set')
```

- Generates performance comparison charts
- Filters by MLflow tags
- Cleans model names for consistent reporting

#### Router Validation (`validation_per_router_results.py`)

Evaluates router decision quality:

```python
validate_experiment("matrix_router_evaluation_multi")
```

- Analyzes model selection distribution
- Calculates efficiency scores (accuracy vs cost/speed trade-offs)
- Compares against baseline individual model performance
- Provides decision quality metrics

## Data Structure

### Test Sets (`test_sets/`)

- **`comprehensive_qa.json`**: Complete test suite with 2000+ query-answer pairs
  - Covers 8+ datasets (academic, atis, broker, car_dealership, derm_treatment, etc.)
  - Includes SQL queries, expected answers, dataset labels, and categories
  - Categories: group_by, order_by, joins, aggregations, filtering

- **`instruct_advanced_postgres.csv`**: Advanced PostgreSQL test cases
- **`questions_gen_postgres.csv`**: Generated question variations

### Generated Data (`../data/`)

- **`model_arena_results.csv`**: Arena competition results (top-2 models per query)
- **`model_arena_results_matrix_router.csv`**: Extended arena results for matrix router
- **`chatbot_arena_inmemory_vectorstore.pkl`**: Embedded arena results for RAG router
- **`database_data/`**: Source databases for SQL queries
- **`llm_examples/`**: Example prompts and responses

### MLflow Runs (`../mlruns/`)

Stores experiment tracking data:
- Run parameters and metrics
- Model artifacts and charts
- Execution logs and traces

## Source Code Reference

### Agents (`../src/agents/`)
- `sql_agent.py`: Base SQL agent with tool execution
- `sql_agent_matrix_router.py`: SQL agent with matrix-based routing
- `sql_agent_rag_router.py`: SQL agent with RAG-based routing
- `supervisor_agent.py`: Supervisor that coordinates model selection

### Routers (`../src/routers/`)
- `matrix_router.py`: ML-based routing logic and model training
- `rag_router.py`: Semantic search routing implementation

### Tools (`../src/tools/`)
- PostgreSQL execution and schema inspection tools
- Cost tracking and logging utilities

### Models (`../src/models/`)
- Unified interface for OpenAI, Anthropic, Google Vertex AI, and local models
- Cost calculation per provider

## Common Utilities (`common/`)

- **`evaluation.py`**: Core evaluation pipeline
- **`plot_utils.py`**: Visualization helpers for MLflow
- **`retry_utils.py`**: Exponential backoff for API calls

## Configuration

- **`configs/llm_model_config.json`**: Available models and their parameters
- **`configs/llm_model_descriptions.json`**: Model capability descriptions for supervisor

## Output

Results are logged to MLflow and can be viewed at `http://localhost:5000`:
- Experiment runs with full metric history
- Performance visualizations
- Model comparison dashboards
- Cost and latency analysis

Validation plots and metrics are also saved to `output/` directory.
