# BenMed

**AI-powered medical report simplification for patients — clinically accurate, plain-language summaries with inline citations and validation.**

---

## Overview

BenMed is a full-stack medical NLP platform built during the Hack-4-Health 2026 hackathon at the Medical College of Wisconsin. It enables patients to upload complex medical documents (PDF or plain text) and receive accessible, plain-language summaries while preserving clinical accuracy through a rigorous, multi-stage validation pipeline.

The system chunks reports into semantic sections, simplifies each in parallel using Google Gemini, retrieves peer-reviewed sources via Exa AI to produce inline citations, and then runs every chunk through seven specialized validators before delivery. Chunks that fail all repair attempts are flagged for clinician review rather than surfaced as-is. The result is a side-by-side view of original and simplified content, a medical glossary, and citation-backed patient education material.

---

## Key Features

- **PDF and plain-text upload** — accepts `.pdf` and `.txt` medical reports up to 10 MB
- **Parallel chunk processing** — document sections are simplified concurrently using an async thread-pool architecture with configurable worker limits
- **7-validator safety pipeline** — numeric consistency, UMLS concept grounding, NLI contradiction detection, negation/laterality preservation, recommendation policy enforcement, concept safety recall, and empty-chunk filtering
- **Automatic repair loops** — chunks that fail validation are sent back to Gemini with a structured repair prompt; after up to 3 iterations, unresolved chunks are flagged `needs_review`
- **Citation-backed simplification** — Exa AI retrieves relevant peer-reviewed sources and clinical guidelines; inline citations are embedded directly in the simplified text
- **Bio-ClinicalBERT semantic similarity** — cosine similarity between original and simplified embeddings verifies meaning preservation independent of keyword matching
- **UMLS Metathesaurus integration** — 3.4M+ medical concepts and 8M+ synonyms indexed in OpenSearch for real-time medical terminology grounding
- **Medical glossary generation** — key terms extracted and defined for patient education
- **Dual-view results interface** — React frontend presents original alongside simplified text, a glossary panel, cited sources, and a physician review mode
- **MLflow experiment tracking** — evaluation metrics are logged for offline analysis

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19, TypeScript, Vite, React Router, react-markdown |
| **Backend** | FastAPI, Uvicorn, Python 3.12+, Pydantic v2 |
| **AI / LLM** | Google Gemini (`google-genai`), Exa AI (`exa-py`) |
| **Medical NLP** | Bio-ClinicalBERT (`sentence-transformers`), medspaCy, spaCy, scispaCy, HuggingFace Transformers |
| **Semantic Search** | OpenSearch (`opensearch-py`) with UMLS Metathesaurus index |
| **PDF Parsing** | PyMuPDF, pypdf, PyPDF2 |
| **Evaluation** | MLflow, BERTScore, scikit-learn |
| **Package Management** | `uv` (backend), `bun` (frontend) |

---

## Architecture Overview

```
Upload (PDF / TXT)
       │
       ▼
  Text Extraction
  (PyMuPDF / UTF-8)
       │
       ▼
  ChunkerService
  (semantic section splitting)
       │
       ▼
  Parallel Chunk Processing (asyncio + ThreadPoolExecutor)
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  1. ExaVerifier.search_sources()                    │
  │     → Exa AI retrieves peer-reviewed sources        │
  │                                                     │
  │  2. SimplifierService.simplify()                    │
  │     → Gemini generates plain-language text          │
  │     → UMLS concepts extracted via OpenSearch        │
  │     → Inline citations inserted from Exa sources    │
  │                                                     │
  │  3. ValidationOrchestrator.validate()               │
  │     ├── EmptyChunkValidator                         │
  │     ├── NumericValidator                            │
  │     ├── UMLSGroundingValidator                      │
  │     ├── ContradictionNLIValidator                   │
  │     ├── RecommendationPolicyValidator               │
  │     └── ConceptSafetyValidator                      │
  │                                                     │
  │     pass → deliver chunk                            │
  │     fail → repair_with_validation_errors()          │
  │          → re-validate (up to 3 iterations)         │
  │          → still fail → needs_review = true         │
  │                                                     │
  │  4. VerifierService.verify_similarity()             │
  │     → Bio-ClinicalBERT cosine similarity score      │
  └─────────────────────────────────────────────────────┘
       │
       ▼
  GlossaryService  →  medical term definitions
       │
       ▼
  API Response  →  React frontend
  (simplified chunks, citations, glossary, needs_review flags)
```

The pipeline supports two processing modes: **parallel** (default, using `asyncio.gather` with a semaphore-limited `ThreadPoolExecutor`) and **sequential** (automatic fallback when called from within a running event loop context).

---

## Getting Started

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for Python package management
- [`bun`](https://bun.sh/) for frontend package management
- A running OpenSearch instance (for UMLS grounding)
- API keys: Google Gemini, Exa AI

### Environment Variables

Create a `.env` file in `api/` with the following:

```env
GOOGLE_API_KEY=your_gemini_api_key
EXA_AI_API_KEY=your_exa_api_key
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### Backend (FastAPI)

```bash
cd api

# Install dependencies
uv sync

# Start the development server
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

#### Load UMLS into OpenSearch (first-time setup)

```bash
cd api
uv run python scripts/load_umls_to_opensearch.py
```

#### Run evaluations

```bash
cd api
uv run run-evals
```

### Frontend (React / Vite)

```bash
cd client

# Install dependencies
bun install

# Start the development server
bun run dev
```

The frontend will be available at `http://localhost:5173`.

#### Build for production

```bash
bun run build
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API info and available endpoints |
| `POST` | `/process-report` | Upload a `.txt` or `.pdf` file for simplification |
| `GET` | `/simplify?test=...` | Simplify a text string directly (development use) |

---

## Project Structure

```
├── api/
│   ├── clients/          # External service clients (Gemini, OpenSearch)
│   ├── evaluators/       # Offline evaluation modules (NLI, semantic, readability, etc.)
│   ├── models/           # Pydantic schemas for API and pipeline data models
│   ├── parser/           # Document parsing and LLM-based structured extraction
│   ├── scripts/          # Utility scripts (UMLS loader, eval runner, dataset tools)
│   ├── services/         # Core business logic (chunker, simplifier, verifier, glossary, etc.)
│   ├── utils/            # Shared helpers (PDF parser, NLP utilities, file handler)
│   ├── validators/       # Validation modules and orchestrator
│   ├── pipeline.py       # Main processing pipeline (parallel and sequential modes)
│   └── main.py           # FastAPI application entry point
└── client/
    └── src/
        ├── components/   # React components (FileUpload, Results, DoctorView, Glossary, etc.)
        └── utils/        # Frontend utilities (document storage, user storage)
```

---

## Hackathon Context

BenMed was built at **Hack-4-Health 2026**, hosted by the Medical College of Wisconsin in January 2026, over a 3-day hackathon sprint.

| | |
|---|---|
| **Event** | Hack-4-Health 2026 — Medical College of Wisconsin |
| **Date** | January 2026 |
| **Duration** | 3 days |
| **Team** | Austin Koske — [austinkoske.com](https://austinkoske.com) |

The project was scoped and delivered end-to-end within the hackathon timeline: infrastructure setup (OpenSearch + UMLS indexing), full pipeline implementation, multi-validator architecture, frontend, and Exa-powered citation retrieval were all completed within 72 hours.
