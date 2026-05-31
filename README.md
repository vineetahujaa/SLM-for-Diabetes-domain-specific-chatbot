# Diabetes Care Chat

A local diabetes-care chatbot built with **FastAPI**, a local **Gemma 270M IT Q8_0 GGUF** model, **RAG over PDFs**, medical guardrails, session memory, and feedback storage.

The app runs the model locally with `llama-cpp-python`. It can answer general diabetes and health questions directly, or use uploaded PDFs as context in RAG mode.

> Medical safety note: this project is for educational and triage-style guidance only. It must not be used for diagnosis, prescribing, insulin dose changes, or emergency decision-making.

---

## What is included in this repo

This repo intentionally keeps only the original project files plus minimal GitHub hygiene files.

```text
.
├── app.py                  # FastAPI app, routes, SSE streaming, rate limiting
├── config.py               # Environment-based configuration
├── generator.py            # Gemma model loading, direct prompt, RAG prompt, topic classifier
├── guardrail.py            # Bedrock input guardrail, medical triage, output policy
├── indexing.py             # FAISS/BM25 index build/load/manifest logic
├── ingestion.py            # PDF loading and chunking
├── retrieval.py            # Hybrid retrieval and CrossEncoder reranking
├── storage.py              # Session memory and SQLite feedback storage
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Test/dev dependencies
├── requirements-finetune.txt # Gemma 270M IT fine-tuning dependencies
├── scripts/
│   ├── build_index.py      # Manual RAG index builder
│   └── fine_tune_gemma_270m_it.py # LoRA SFT training + merge script
├── templates/
│   └── index.html          # Frontend UI
├── tests/                  # Existing tests
├── data/.gitkeep           # Empty folder for user PDFs
├── saved_indexes/.gitkeep  # Empty folder for generated RAG indexes
├── .env.example            # Same config layout as local .env, secrets left blank
└── .gitignore              # Excludes .env, model files, PDFs, DBs, indexes, cache files
```

Not included in GitHub package:

```text
.env
*.gguf
feedback.sqlite3
data/*.pdf
saved_indexes/*
train_mixed.csv
qa_cleaned_shuffled.csv
gemma-diabetes*/
__pycache__/
.pytest_cache/
.DS_Store
__MACOSX/
.claude/
```

These are local/private/generated files and should not be committed.

---

## Main features

- Local **Gemma 270M IT Q8_0 GGUF** inference through `llama-cpp-python`
- Fine-tuning script for **google/gemma-3-270m-it** using LoRA SFT
- Two chat modes:
  - **Model + Guardrail**: answer directly from the local model
  - **Model + RAG**: retrieve PDF context, answer using that context, then show sources
- Server-Sent Events streaming for live token output
- Input safety using **Amazon Bedrock Guardrails** when enabled
- Offline medical safety checks for high-risk diabetes symptoms and glucose values
- Output safety using `MedicalResponsePolicy`
- RAG pipeline using PDF ingestion, chunking, FAISS, BM25, and CrossEncoder reranking
- Session memory with TTL and max-session cleanup
- Feedback storage in SQLite
- PDF upload from UI or API
- Manual and API-based RAG index rebuild
- Basic health/readiness endpoints
- Tests for app behavior, storage, and guardrails

---

## Modes

### 1. Model + Guardrail mode

This is the direct model mode.

Flow:

```text
User question
  -> input guardrail check
  -> high-risk medical triage check
  -> local Gemma generation
  -> output medical policy validation
  -> streamed answer
```

Use this when the user is asking a general diabetes or health question that does not need PDF context.

### 2. Model + RAG mode

This mode answers using uploaded PDFs.

Flow:

```text
User question
  -> input guardrail check
  -> high-risk medical triage check
  -> retrieve relevant PDF chunks
  -> rerank chunks
  -> inject context into Gemma prompt
  -> output medical policy validation
  -> streamed answer + source pages
```

Use this when the answer should come from PDFs placed in `data/` or uploaded through the app.

---

## Guardrail strategy

The project uses multiple safety layers. They are intentionally placed both before and after model generation.

### Layer 1: Input safety with Amazon Bedrock Guardrails

File: `guardrail.py`

Function: `check_unsafe_input()`

When `BEDROCK_MODERATION_ENABLED=true`, the app sends the user question to Amazon Bedrock Guardrails before calling the local model.

The Bedrock layer is meant to block unsafe categories such as:

- self-harm
- sexual content involving minors
- violence
- hate or abusive content

If Bedrock returns `GUARDRAIL_INTERVENED`, the request is blocked before the model runs.

For self-harm style content, the app returns a crisis-support style message instead of model output.

If Bedrock is disabled, the app can still run offline, but cloud moderation will not be active.

### Layer 2: High-risk diabetes and medical triage

File: `guardrail.py`

Function: `check_high_risk_medical_input()`

This layer checks the user question for urgent medical risk patterns before model generation.

Examples of high-risk signals:

- very high glucose values
- very low glucose values
- DKA or ketone-related danger signs
- chest pain
- seizure
- unconsciousness
- severe vomiting/dehydration patterns

When urgent content is detected, the app does not generate a normal chatbot answer. It returns a safety response telling the user to seek urgent medical help.

This is important because the model should not casually answer emergency medical situations.

### Layer 3: System prompt safety

File: `generator.py`

The system prompt tells the model to:

- focus on diabetes care
- answer only diabetes, health, greeting, or valid follow-up questions
- keep answers short, practical, and safe
- not diagnose
- not prescribe
- not adjust medication or insulin doses
- recommend clinician/emergency care for urgent symptoms
- refuse unrelated topics

This reduces unsafe behavior during generation, but the app does not rely only on the prompt.

### Layer 4: Output validation with `MedicalResponsePolicy`

File: `guardrail.py`

Class: `MedicalResponsePolicy`

After the model finishes generating, the complete answer is checked before it is accepted.

The output policy checks things like:

- whether the question is diabetes/health related
- whether the question is a valid follow-up to a previous medical answer
- whether the answer starts with an unwanted greeting
- whether off-topic answers are too long
- whether the model answered an unrelated topic instead of refusing

For topic classification, the policy first checks known diabetes and general-health keywords. If needed, it uses the loaded Gemma model as a tiny classifier that answers only `YES` or `NO`.

If the answer fails validation, the frontend receives a `replace` event and the streamed text is replaced with the refusal line:

```text
I'm your virtual healthcare professional, and I can only assist with diabetes and health-related questions.
```

### Layer 5: Rate limiting and request size limits

File: `app.py`

The API has per-IP sliding-window rate limiting.

Relevant env variables:

```env
RATE_LIMIT_ENABLED=true
RATE_LIMIT_MAX_REQUESTS=30
RATE_LIMIT_WINDOW_SECONDS=60
MAX_QUESTION_CHARS=2000
MAX_FEEDBACK_CHARS=12000
MAX_SUGGESTION_CHARS=2000
```

This helps reduce abuse, accidental overload, and very large prompts.

---

## RAG strategy

The RAG pipeline is designed for local PDF-based diabetes reference documents.

### Step 1: PDF ingestion

File: `ingestion.py`

PDFs are loaded from:

```env
DATA_DIR=./data
```

The app uses PyMuPDF to read PDF pages. Each page is converted into text and stored with metadata such as source filename and page number.

### Step 2: Chunking

File: `ingestion.py`

Documents are split into overlapping chunks.

Default values:

```env
CHUNK_SIZE=400
CHUNK_OVERLAP=70
```

The overlap helps preserve context between chunks so that important medical statements are not cut off too aggressively.

### Step 3: Dense vector index with FAISS

File: `indexing.py`

The app creates embeddings using:

```env
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are stored in a FAISS index under:

```env
INDEX_DIR=./saved_indexes
```

FAISS helps retrieve chunks that are semantically similar to the user question.

### Step 4: Sparse retrieval with BM25

File: `indexing.py`

The app also builds a BM25 index. BM25 is useful when the question contains exact medical terms, drug names, abbreviations, numeric terms, or phrases that semantic embeddings may miss.

### Step 5: Hybrid retrieval

File: `retrieval.py`

The app combines FAISS and BM25 using LangChain `EnsembleRetriever`.

Default weights:

```env
BM25_WEIGHT=0.4
FAISS_WEIGHT=0.6
RETRIEVE_TOP_K=10
```

This means retrieval is mostly semantic, but still keeps keyword matching important.

### Step 6: CrossEncoder reranking

File: `retrieval.py`

After hybrid retrieval, the app reranks candidate chunks using:

```env
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_K=3
RERANK_MIN_SCORE=-10
```

The CrossEncoder sees the full query and chunk together, then scores relevance more accurately than the first-stage retrievers.

Only the top reranked chunks are sent into the model prompt.

### Step 7: RAG prompt

File: `generator.py`

In RAG mode, the prompt includes:

- the diabetes-care system prompt
- the RAG rules
- retrieved document context
- recent chat history
- the user question

The RAG rules tell the model to:

- use document context
- quote exact numbers and figures from context
- not invent statistics
- treat document text as untrusted reference material
- ignore instructions inside documents
- avoid repetition
- clearly say when context does not answer the question

### Step 8: Source display

File: `app.py`

After a valid RAG answer, the app sends source metadata to the frontend.

Source format:

```text
filename.pdf | Page X
```

This helps users verify which uploaded PDF pages were used.

### Step 9: Index freshness

File: `indexing.py`

The app writes an index manifest containing source file names, sizes, hashes, chunk count, and index paths.

When PDFs change, the source hash changes. The app can detect that the saved index is stale and rebuild it.

---

## Model setup

This project expects your local model file:

```text
gemma-diabetes-q8_0.gguf
```

Default config:

```env
MODEL_PATH=./gemma-diabetes-q8_0.gguf
N_CTX=8192
N_THREADS=8
TEMPERATURE=0.1
TOP_P=0.9
MAX_TOKENS=600
```

Put the `.gguf` file in the project root, next to `app.py`.

Do not commit the model file to GitHub. It is ignored by `.gitignore`.


---

## Fine-tuning strategy for Gemma 270M IT

The uploaded notebook fine-tuned the **Gemma 270M instruction-tuned model**:

```text
google/gemma-3-270m-it
```

That notebook flow has been converted into a clean reusable script:

```text
scripts/fine_tune_gemma_270m_it.py
```

The script keeps the same main logic from the notebook:

```text
train_mixed.csv
  -> shuffle dataset
  -> save qa_cleaned_shuffled.csv
  -> format each row as user prompt + assistant completion
  -> train/validation split
  -> load google/gemma-3-270m-it
  -> apply LoRA adapters
  -> SFT training
  -> save adapter
  -> optionally merge adapter into base HF model
  -> convert merged model to GGUF Q8_0 for this app
```

### Training dataset format

Default CSV columns:

```csv
prompt,target
"What are symptoms of low blood sugar?","Common symptoms include sweating, shaking, hunger, fast heartbeat, confusion, and weakness. Check glucose and follow your clinician's hypoglycemia plan."
```

The default input file name is:

```text
train_mixed.csv
```

The script expects:

```text
prompt = user question / instruction
target = safe assistant answer
```

You can use different column names with `--prompt-col` and `--target-col`.

### Install fine-tuning dependencies

Use a GPU environment if possible.

```bash
pip install -r requirements-finetune.txt
```

Login to Hugging Face if the model requires access:

```bash
huggingface-cli login
```

Or pass a token:

```bash
export HF_TOKEN=your_huggingface_token
```

### Run fine-tuning

From the project root:

```bash
python scripts/fine_tune_gemma_270m_it.py \
  --train-csv train_mixed.csv \
  --prompt-col prompt \
  --target-col target \
  --merge
```

Default training settings are based on the original notebook:

```text
model-id: google/gemma-3-270m-it
LoRA r: 16
LoRA alpha: 32
LoRA dropout: 0.05
target modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
epochs: 3
learning rate: 2e-4
batch size: 8
gradient accumulation: 4
max length: 512
scheduler: cosine
warmup ratio: 0.05
```

Outputs:

```text
gemma-diabetes/             # training checkpoints
gemma-diabetes-final/       # LoRA adapter
gemma-diabetes-merged-hf/   # merged Hugging Face model, if --merge is used
qa_cleaned_shuffled.csv     # shuffled cleaned dataset
```

These outputs are local training artifacts and should not be committed.

### Convert merged model to GGUF Q8_0

The FastAPI app loads a GGUF file using `llama-cpp-python`, so after fine-tuning and merging, convert the merged HF model to GGUF and quantize it.

Example with llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
cd ..

python llama.cpp/convert_hf_to_gguf.py \
  ./gemma-diabetes-merged-hf \
  --outfile ./gemma-diabetes-f16.gguf

./llama.cpp/build/bin/llama-quantize \
  ./gemma-diabetes-f16.gguf \
  ./gemma-diabetes-q8_0.gguf \
  Q8_0
```

Then keep this file in the project root:

```text
gemma-diabetes-q8_0.gguf
```

The app will load it through:

```env
MODEL_PATH=./gemma-diabetes-q8_0.gguf
```

Do not push the `.gguf` file to GitHub.

### Fine-tuning safety strategy

For this healthcare chatbot, the fine-tuning dataset should teach the model to answer safely, not just correctly.

Recommended target-answer rules:

- keep diabetes answers short and practical
- do not diagnose the user
- do not prescribe medicine
- do not change insulin or medication doses
- include clinician advice for uncertain cases
- include emergency advice for severe symptoms
- refuse unrelated non-health questions
- avoid giving exact treatment plans without professional supervision
- avoid overconfident claims

Fine-tuning improves style and domain behavior, but it does not replace runtime guardrails. This repo still uses Bedrock input moderation, high-risk medical triage, system prompt safety, and output validation.


---

## Environment setup

Create your local `.env` from the example:

```bash
cp .env.example .env
```

Then edit `.env` if needed.

Important values:

```env
MODEL_PATH=./gemma-diabetes-q8_0.gguf
DATA_DIR=./data
INDEX_DIR=./saved_indexes
FEEDBACK_DB=./feedback.sqlite3
BEDROCK_MODERATION_ENABLED=true
AWS_REGION=ap-south-1
BEDROCK_GUARDRAIL_ID=6w4ikb8z66kb
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

For GitHub, keep real AWS keys only in your local `.env`, never in `.env.example`.

---

## Installation

Use Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For fine-tuning only:

```bash
pip install -r requirements-finetune.txt
```

For tests:

```bash
pip install -r requirements-dev.txt
```

---

## Running the app

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or:

```bash
python app.py
```

Open:

```text
http://localhost:8000
```

---

## Using PDFs for RAG

### Option 1: Add PDFs manually

Place PDF files inside:

```text
data/
```

Then build the index:

```bash
python scripts/build_index.py
```

### Option 2: Upload PDFs from the UI

Open the app, switch to **Model + RAG**, and upload a PDF using the UI.

### Option 3: Upload PDFs by API

```bash
curl -X POST http://localhost:8000/api/upload_document \
  -F "file=@your-document.pdf"
```

After upload, the app marks RAG as stale. The next RAG query or manual rebuild can refresh the index.

---

## Rebuilding the RAG index

### Script

```bash
python scripts/build_index.py
```

### API

```bash
curl -X POST http://localhost:8000/api/rebuild_index
```

If `ADMIN_TOKEN` is set:

```bash
curl -X POST http://localhost:8000/api/rebuild_index \
  -H "X-Admin-Token: your-token"
```

In production, `ADMIN_TOKEN` is required.

---

## API endpoints

### `GET /`

Returns the web UI.

### `GET /health`

Simple health check.

Example response:

```json
{
  "status": "ok",
  "env": "development",
  "sessions": 1
}
```

### `GET /ready`

Readiness check for model file, feedback DB, and RAG index.

Example response:

```json
{
  "status": "ready",
  "model_present": true,
  "feedback_db": true,
  "rag_index_current": true,
  "rag_error": null
}
```

### `POST /chat_stream`

Streams chat output as Server-Sent Events.

Request body:

```json
{
  "question": "What are symptoms of low blood sugar?",
  "mode": "guardrail"
}
```

Modes:

```text
guardrail
rag
```

The frontend sends/uses `X-Session-Id` for memory.

SSE event types include:

```text
ttft
searching
generating
token
sources
valid
replace
tps
status
```

### `POST /feedback`

Stores user feedback in SQLite.

Request body:

```json
{
  "session_id": "default",
  "question": "...",
  "answer": "...",
  "feedback": "up",
  "suggestion": "optional text"
}
```

### `POST /api/clear_memory`

Clears the current session memory.

### `POST /end_session`

Ends a session by session ID.

### `POST /api/rebuild_index`

Rebuilds RAG indexes from PDFs.

### `GET /api/documents`

Lists uploaded PDFs.

### `POST /api/upload_document`

Uploads a PDF into `DATA_DIR`.

---

## Session memory strategy

File: `storage.py`

The app stores short-term session memory in memory, not in a database.

Config:

```env
SESSION_TTL_SECONDS=7200
MAX_SESSIONS=1000
```

The memory store keeps:

- recent user/assistant messages
- last question
- last response
- whether the last response passed the medical policy
- last update timestamp

This is used for valid follow-up detection. For example, after a diabetes answer, a user can ask "tell me more" and the app can treat it as a medical follow-up.

---

## Feedback strategy

File: `storage.py`

Feedback is written to SQLite:

```env
FEEDBACK_DB=./feedback.sqlite3
```

The table stores:

- session ID
- question
- answer
- feedback value
- optional suggestion
- timestamp
- metadata JSON

The database is local runtime state and is ignored by Git.

---

## Security strategy

### What is protected

- `.env` is ignored
- AWS keys are not included in `.env.example`
- `.gguf` model files are ignored
- uploaded PDFs are ignored
- SQLite feedback DB is ignored
- generated FAISS/BM25 indexes are ignored
- Python/cache/Mac files are ignored

### Production recommendations

Before production deployment:

```env
APP_ENV=production
DEBUG=false
ALLOWED_HOSTS=your-domain.com
CORS_ORIGINS=https://your-domain.com
ADMIN_TOKEN=a-long-random-secret
BEDROCK_MODERATION_ENABLED=true
```

Also recommended:

- use IAM role or secret manager instead of hardcoding AWS keys
- use HTTPS behind a reverse proxy
- monitor Bedrock moderation failures
- keep only trusted indexes when `ALLOW_DANGEROUS_INDEX_DESERIALIZATION=true`
- run one model worker per machine unless you have enough RAM for multiple copies
- review medical/legal disclaimers for your use case

---

## Tests

Install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

Run:

```bash
pytest -q
```

---

## GitHub upload steps

From inside the project folder:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Before pushing, check that private files are not staged:

```bash
git status --short
```

You should not see:

```text
.env
*.gguf
feedback.sqlite3
data/*.pdf
saved_indexes/*
```

---

## Troubleshooting

### Model file not found

Make sure the model exists at:

```text
./gemma-diabetes-q8_0.gguf
```

Or update:

```env
MODEL_PATH=/full/path/to/your/model.gguf
```

### PDF upload fails

Only `.pdf` files are accepted.

Also make sure dependencies are installed:

```bash
pip install -r requirements.txt
```

### RAG says no context found

Check that PDFs exist in `data/`, then rebuild:

```bash
python scripts/build_index.py
```

### Bedrock moderation is not working

Check:

```env
BEDROCK_MODERATION_ENABLED=true
AWS_REGION=ap-south-1
BEDROCK_GUARDRAIL_ID=your-guardrail-id
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

Also verify that your AWS identity has permission to call Bedrock Guardrails.

### App is slow

Local CPU inference can be slow. Try:

```env
N_THREADS=number-of-cpu-threads
MAX_TOKENS=lower-number
N_CTX=lower-number-if-needed
```

---

## Project summary

This app is a local diabetes-care assistant with two answer paths:

```text
Direct mode: user -> guardrails -> Gemma -> output policy -> answer
RAG mode:    user -> guardrails -> PDF retrieval -> Gemma -> output policy -> answer + sources
```

The core idea is simple: keep generation local, use PDFs only as grounded context, and use guardrails before and after generation so the chatbot stays focused on diabetes and general health.
