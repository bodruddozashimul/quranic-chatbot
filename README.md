# Quran RAG

A local Retrieval-Augmented Generation (RAG) demo for asking questions about Quranic verses. The stack uses ChromaDB for vector storage, multilingual sentence-transformer embeddings for retrieval, Ollama for local language-model inference, FastAPI for the backend, and Streamlit for the user interface.

The supplied Docker Compose configuration is tuned for a Windows 11 machine with approximately 8 GB RAM. In the default configuration, Ollama runs natively on Windows rather than inside Docker so that the model does not compete with the containers for the limited WSL2 memory allocation.

## Architecture

```text
Browser
   |
   v
Streamlit frontend :8501
   |
   v
FastAPI backend :8000
   |                    \
   v                     v
ChromaDB :8002       Ollama on Windows :11434
   ^
   |
Dataset ingestion (one-time / when rebuilding the dataset)
```

### Services

| Service | Purpose | Host port |
|---|---|---:|
| `chromadb` | Persistent vector database | `8002` |
| `ingest` | Builds/downloads the Quran dataset, creates embeddings and loads ChromaDB | — |
| `backend` | FastAPI application; connects to ChromaDB and Ollama | `8000` |
| `frontend` | Streamlit user interface | `8501` |
| `ollama` | Optional containerized Ollama service | `11434` |

The ingestion service is configured to download three editions and ingest 6,236 ayahs. The exact edition details and dataset source are determined by the ingestion implementation.

## Prerequisites

For the default Windows-host Ollama setup:

- Windows 11 with WSL2 enabled.
- Docker Desktop using the WSL2 backend and Docker Compose.
- Ollama installed and running on Windows.
- Sufficient free disk space for downloaded models, embeddings and persistent data.

The Compose file applies container memory limits. Actual resource use also depends on the selected model, Docker Desktop/WSL2 allocation and the host machine.

## Quick start (Windows 11)

Run these commands from PowerShell in the repository root.

### 1. Clone the repository

```powershell
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Create the local environment file

```powershell
Copy-Item .env.example .env
```

Review `.env` and adjust the settings if required. Do not commit your real `.env` file.

### 3. Install and start Ollama on Windows

Install Ollama from its official website, start it, then download the default model:

```powershell
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama list
```

The default `.env` uses:

```dotenv
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

This address is intended for containers to reach Ollama running on the Windows host. Make sure Ollama is running before sending questions to the application.

### 4. Build and start the Docker stack

```powershell
docker compose up --build -d
```

This starts ChromaDB, the ingestion job, the FastAPI backend and the Streamlit frontend. The ingestion container is configured to run once and exit after processing. On first startup, dataset downloads and embedding generation may take some time.

### 5. Open the application

- Streamlit UI: http://localhost:8501
- FastAPI service: http://localhost:8000
- FastAPI interactive API documentation (if enabled by the backend): http://localhost:8000/docs
- ChromaDB: http://localhost:8002

## Useful Docker commands

View running containers:

```powershell
docker compose ps
```

Follow all service logs:

```powershell
docker compose logs -f
```

Follow a particular service:

```powershell
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f chromadb
docker compose logs -f ingest
```

Stop the stack:

```powershell
docker compose down
```

Stop the stack and remove its containers/network (persistent bind-mounted data under `volumes/` is retained):

```powershell
docker compose down
```

Rebuild application images after code changes:

```powershell
docker compose up --build -d
```

To rerun ingestion, first ensure ChromaDB is running, then run:

```powershell
docker compose run --build --rm ingest
```

Only rerun ingestion when you intend to rebuild/reload the dataset. Review the ingestion implementation before doing so against an existing collection.

## Optional: run Ollama inside Docker

The Compose file includes an optional `ollama` service under the `bundled-ollama` profile. This is not the recommended configuration for an 8 GB machine because the model and the other containers share Docker/WSL2 memory.

To use it:

1. Change the following setting in `.env`:

   ```dotenv
   OLLAMA_BASE_URL=http://ollama:11434
   ```

2. Start the stack with the optional profile:

   ```powershell
   docker compose --profile bundled-ollama up --build -d
   ```

The selected model must also be available to the containerized Ollama instance. The model data is persisted under `volumes/ollama`.

## Configuration reference

All backend settings are read from `.env` by the Compose `backend` service.

| Variable | Example default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `qwen2.5:3b-instruct-q4_K_M` | Local Ollama model name |
| `TEMPERATURE` | `0.2` | Generation temperature |
| `NUM_PREDICT` | `220` | Maximum number of predicted tokens |
| `NUM_CTX` | `2048` | Model context window setting |
| `LLM_TIMEOUT` | `180` | LLM request timeout |
| `EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `EMBED_THREADS` | `2` | Embedding worker threads for the backend |
| `TOP_K` | `3` | Number of retrieved results |
| `SIM_THRESHOLD` | `0.35` | Similarity threshold for the out-of-scope gate |
| `USE_HYBRID` | `false` | Enables lexical/vector retrieval fusion |
| `BM25_WEIGHT` | `0.3` | BM25 weight when hybrid retrieval is enabled |
| `COLLECTION_NAME` | `quran_ayahs` | ChromaDB collection name |

The ingestion service separately sets `EMBED_THREADS=4` in `docker-compose.yml`.

### Retrieval tuning

- `SIM_THRESHOLD`: increase cautiously if unrelated verses are being retrieved; decrease cautiously if valid questions are rejected. Evaluate changes with the project's evaluation dataset.
- `TOP_K`: controls how many retrieved results are supplied.
- `USE_HYBRID` and `BM25_WEIGHT`: control optional lexical/vector fusion. Keep hybrid retrieval disabled unless you intend to test and evaluate it.

## Persistent data

The Compose file uses bind mounts under `volumes/`:

- `volumes/chromadb/` — persistent ChromaDB data.
- `volumes/data/` — downloaded/generated dataset files.
- `volumes/ollama/` — model data when using the optional containerized Ollama profile.

Back up these directories before deleting or rebuilding data. Do not commit runtime data, downloaded datasets, model files, or database files to Git unless you have explicitly reviewed their size and redistribution permissions.

## Repository hygiene and security

- Commit `.env.example`; keep `.env` local and untracked.
- Keep `volumes/`, model files, caches, virtual environments and generated data out of Git.
- Never commit passwords, API keys, tokens, private datasets or machine-specific settings.
- Review the repository for secrets before the first push. If a secret has already been committed, removing it in a later commit is not sufficient; rotate the exposed secret and clean the Git history.
- The supplied `.dockerignore` excludes `volumes/`, `__pycache__/` and `*.pyc` from Docker build contexts. `.dockerignore` does not replace `.gitignore`.

## Project structure

The Compose file expects these directories:

```text
.
├── backend/              # FastAPI backend
├── data/                 # Dataset build / ingestion image
├── frontend/             # Streamlit frontend
├── volumes/              # Persistent runtime data (not for Git)
├── .env                  # Local configuration (not for Git)
├── .env.example          # Shareable configuration template
├── .dockerignore
├── .gitignore
└── docker-compose.yml
```

This is the structure referenced by `docker-compose.yml`; add any other project directories or files that exist in your repository.

## Troubleshooting

**Backend cannot connect to Ollama**

- Confirm Ollama is running on Windows.
- Run `ollama list` and verify that the configured model is present.
- Confirm `.env` contains `OLLAMA_BASE_URL=http://host.docker.internal:11434` for the default setup.
- Inspect backend logs with `docker compose logs -f backend`.

**The first startup takes a long time**

- Dataset downloads and embedding generation happen during ingestion. Inspect `docker compose logs -f ingest`.
- Check available RAM and disk space.

**The UI cannot reach the backend**

- Confirm all services are running with `docker compose ps`.
- Inspect `frontend` and `backend` logs.
- The frontend container is configured to reach the backend at `http://backend:8000` on the Compose network.

## License and data attribution

Add the project's chosen software license and the required attribution/licensing information for each Quran translation or dataset edition before publishing the repository publicly. Do not assume that downloaded translations are unrestricted for redistribution.
