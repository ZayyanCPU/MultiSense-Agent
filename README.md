<div align="center">

# 🧠 MultiSense Agent

### Multi-Modal AI Chatbot with WhatsApp Integration

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-FREE_API-FFD21E?style=for-the-badge)](https://huggingface.co/docs/api-inference)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![n8n](https://img.shields.io/badge/n8n-Automation-EA4B71?style=for-the-badge&logo=n8n&logoColor=white)](https://n8n.io)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-Cloud_API-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)](https://developers.facebook.com/docs/whatsapp/cloud-api)

An intelligent multi-modal chatbot that processes **text**, **voice**, **images**, and **PDFs** using RAG (Retrieval Augmented Generation) and n8n workflow automation. Powered by **FREE open-source models** via HuggingFace Inference API. Integrates seamlessly with WhatsApp Business API for real-world conversational AI.

[Architecture](#-architecture) · [Features](#-features) · [Quick Start](#-quick-start) · [API Docs](#-api-endpoints) · [n8n Workflows](#-n8n-workflows)

</div>

---

## 📋 Features

| Modality | Capability | Technology |
|----------|-----------|------------|
| 💬 **Text** | Conversational AI with memory & RAG context | Mistral-7B + ChromaDB |
| 🎤 **Voice** | Speech-to-text transcription → AI response | Whisper-v3 + Mistral-7B |
| 📸 **Image** | Visual analysis & description | BLIP + LLM follow-up |
| 📄 **PDF** | Document ingestion → knowledge base | RAG + LangChain |
| 📱 **WhatsApp** | Full WhatsApp Business integration | Cloud API v21.0 |
| 🔄 **Automation** | Workflow orchestration & routing | n8n |

### Key Highlights
- **RAG Pipeline**: Upload PDFs → automatic chunking → embeddings → semantic search
- **100% FREE AI**: Uses HuggingFace Inference API — no paid API keys required
- **Open-Source Models**: Mistral-7B, Whisper-v3, BLIP, sentence-transformers
- **Conversation Memory**: Per-user session tracking with configurable TTL
- **Multi-modal Routing**: Automatic detection and routing of text/voice/image/document inputs
- **Production Docker Setup**: Multi-stage builds, health checks, non-root containers
- **n8n Workflow Integration**: Pre-built automation workflows for WhatsApp message routing
- **Async Architecture**: Fully async FastAPI with background task processing

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     WhatsApp Users                           │
│              📱 Text | 🎤 Voice | 📸 Image | 📄 PDF         │
└─────────────────────────┬────────────────────────────────────┘
                          │ WhatsApp Cloud API
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    n8n Automation Layer                       │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │   Webhook    │→ │  Message     │→ │  Route by Type    │   │
│  │   Trigger    │  │  Parser      │  │  (Switch Node)    │   │
│  └─────────────┘  └──────────────┘  └─────────┬─────────┘   │
└────────────────────────────────────────────────┼─────────────┘
                          │ HTTP Request to FastAPI
                          ▼
┌──────────────────────────────────────────────────────────────┐
│              FastAPI Backend (MultiSense Agent)               │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │            Multi-Modal Processor                     │     │
│  │  ┌──────┐ ┌───────┐ ┌───────┐ ┌──────────────┐    │     │
│  │  │ Text │ │ Voice │ │ Image │ │  Document/PDF │    │     │
│  │  │      │ │       │ │       │ │              │    │     │
│  │  │ RAG  │ │Whisper│ │ BLIP  │ │  RAG Ingest  │    │     │
│  │  │ +LLM │ │ +LLM  │ │ +LLM  │ │  Pipeline    │    │     │
│  │  └──┬───┘ └───┬───┘ └───┬───┘ └──────┬───────┘    │     │
│  └─────┼─────────┼─────────┼────────────┼────────────┘     │
│        │         │         │            │                    │
│  ┌─────▼─────────▼─────────▼────────────▼──────────────┐    │
│  │          Conversation Memory Service                 │    │
│  │      (Per-user session tracking with TTL)            │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────┬───────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  HuggingFace │  │   ChromaDB   │  │  WhatsApp    │
    │  Inference   │  │  Vector DB   │  │  Cloud API   │
    │  (FREE API)  │  │  Embeddings  │  │  Send/Recv   │
    │  Mistral/    │  │  Storage     │  │  Media DL    │
    │  Whisper/    │  └──────────────┘  └──────────────┘
    │  BLIP/ST     │
    └──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Where to get it | Cost |
|------------|----------------|------|
| **Python 3.11+** | [python.org](https://www.python.org/downloads/) | Free |
| **Docker & Docker Compose** | [docker.com](https://docs.docker.com/get-docker/) | Free |
| **HuggingFace Account** | [huggingface.co/join](https://huggingface.co/join) | Free |
| **HuggingFace API Token** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Free |
| **ngrok** *(optional — for WhatsApp webhooks)* | [ngrok.com](https://ngrok.com/) | Free tier |

---

### Step 1 — Get your FREE HuggingFace API Token

1. Go to **[huggingface.co/join](https://huggingface.co/join)** and create a free account (or log in)
2. Go to **[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)**
3. Click **"Create new token"**
4. Set:
   - **Name**: `MultiSense-Agent`
   - **Type**: `Read` (that's all you need)
5. Click **"Generate"** and **copy the token** — it starts with `hf_...`

> 💡 This token gives you free access to thousands of AI models on HuggingFace — no credit card required.

---

### Step 2 — Clone the Project

```bash
git clone https://github.com/YOUR_USERNAME/MultiSense-Agent.git
cd MultiSense-Agent
```

---

### Step 3 — Configure Environment Variables

```bash
# Copy the template
cp .env.example .env
```

Now open `.env` in any text editor and **paste your HuggingFace token**:

```env
# ── REQUIRED ─────────────────────────────────────────
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # ← paste your token here

# ── AI MODELS (defaults work great, no changes needed) ──
HF_CHAT_MODEL=mistralai/Mistral-7B-Instruct-v0.3
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_WHISPER_MODEL=openai/whisper-large-v3
HF_VISION_MODEL=Salesforce/blip-image-captioning-large

# ── WHATSAPP (only if you want WhatsApp integration) ──
# WHATSAPP_API_TOKEN=your-meta-api-token
# WHATSAPP_PHONE_NUMBER_ID=your-phone-number-id
# WHATSAPP_VERIFY_TOKEN=any-secret-string-you-choose
```

> **That's it — `HF_API_TOKEN` is the only required variable.** Everything else has sensible defaults.

---

### Step 4 — Run the Project

#### Option A: Docker Compose (Recommended)

This starts the entire stack — FastAPI backend, ChromaDB vector database, and n8n automation — in one command:

```bash
# Build and start all 3 services
docker compose up -d --build

# Check everything is running
docker compose ps

# You should see:
#   multisense-agent     running   0.0.0.0:8080->8080/tcp
#   multisense-chromadb  running   0.0.0.0:8000->8000/tcp
#   multisense-n8n       running   0.0.0.0:5678->5678/tcp
```

#### Option B: Run Locally (without Docker)

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start ChromaDB separately (needs Docker or a running instance)
docker run -d -p 8000:8000 chromadb/chroma:0.5.23

# 5. Update .env for local development
#    Set CHROMA_HOST=localhost (instead of "chromadb")

# 6. Run the FastAPI server
python -m app.main
```

---

### Step 5 — Verify It's Working

**Health check:**
```bash
curl http://localhost:8080/health
# → {"status": "healthy", ...}
```

**Open the interactive API docs:**
Open your browser and go to → **http://localhost:8080/docs**

You'll see the Swagger UI with all endpoints ready to test.

---

### Step 6 — Try It Out!

**💬 Send a text message:**
```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?", "session_id": "test-user-1"}'
```

**📄 Upload a PDF to the knowledge base:**
```bash
curl -X POST http://localhost:8080/api/v1/upload \
  -F "file=@path/to/your-document.pdf"
```

**💬 Ask questions about the uploaded PDF:**
```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize the key findings", "session_id": "test-user-1", "use_rag": true}'
```

**🎤 Send a voice message:**
```bash
curl -X POST http://localhost:8080/api/v1/chat/voice \
  -F "audio=@recording.ogg" \
  -F "session_id=test-user-1"
```

**📸 Analyze an image:**
```bash
curl -X POST http://localhost:8080/api/v1/chat/image \
  -F "image=@photo.jpg" \
  -F "caption=What do you see in this image?" \
  -F "session_id=test-user-1"
```

> 💡 **Tip:** You can also do all of this from the Swagger UI at `http://localhost:8080/docs` — no curl needed!

---

### Step 7 — View Logs & Stop

```bash
# View live logs
docker compose logs -f multisense-agent

# Stop all services
docker compose down

# Stop and remove all data (fresh start)
docker compose down -v
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Project info & capabilities |
| `GET` | `/health` | Service health check |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/api/v1/chat` | Text chat with RAG |
| `POST` | `/api/v1/chat/voice` | Voice message processing |
| `POST` | `/api/v1/chat/image` | Image analysis |
| `POST` | `/api/v1/upload` | PDF document ingestion |
| `GET` | `/api/v1/webhook/whatsapp` | WhatsApp webhook verification |
| `POST` | `/api/v1/webhook/whatsapp` | WhatsApp message receiver |
| `GET` | `/api/v1/sessions` | List active sessions |
| `DELETE` | `/api/v1/sessions/{id}` | Clear session memory |
| `GET` | `/api/v1/knowledge-base/stats` | RAG collection statistics |

### Chat Request Example

```json
POST /api/v1/chat
{
  "message": "Explain the key findings from the uploaded research paper",
  "session_id": "user-123",
  "use_rag": true
}
```

### Response Example

```json
{
  "response": "Based on the research paper, the key findings are...",
  "session_id": "user-123",
  "sources": ["research_paper.pdf"],
  "processing_time": 2.34,
  "message_type": "text"
}
```

---

## 🔄 n8n Workflows

The project includes pre-built n8n workflow templates in `n8n/workflows/`:

### WhatsApp Chatbot Workflow
Handles the complete message lifecycle:
1. **Webhook Trigger** → Receives WhatsApp messages
2. **Message Parser** → Extracts message type & content
3. **Type Router** → Routes text/voice/image/document
4. **API Call** → Sends to MultiSense Agent backend
5. **Response Sender** → Sends reply back via WhatsApp

### Document Ingestion Workflow
Automates document processing:
1. **Upload Trigger** → Receives document uploads
2. **Type Validator** → Checks for PDF format
3. **RAG Ingestion** → Processes into vector knowledge base

### Import Workflows
1. Open n8n at `http://localhost:5678`
2. Login with credentials from `.env`
3. Import workflow JSON files from `n8n/workflows/`
4. Configure WhatsApp Business credentials
5. Activate workflows

---

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `multisense-agent` | 8080 | FastAPI backend |
| `chromadb` | 8000 | Vector database |
| `n8n` | 5678 | Workflow automation |

---

## 📁 Project Structure

```
MultiSense-Agent/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Pydantic settings & configuration
│   ├── models.py               # Request/response schemas
│   ├── routes/
│   │   ├── chat.py             # Direct chat API endpoints
│   │   ├── webhook.py          # WhatsApp webhook handlers
│   │   └── health.py           # Health check & utilities
│   └── services/
│       ├── hf_service.py       # HuggingFace API (Chat, Whisper, Vision, Embeddings)
│       ├── rag_engine.py       # RAG pipeline (ChromaDB + HF Embeddings)
│       ├── whatsapp_service.py # WhatsApp Cloud API client
│       ├── memory_service.py   # Conversation memory management
│       └── processor.py        # Multi-modal routing orchestrator
├── n8n/
│   └── workflows/
│       ├── whatsapp_chatbot.json
│       └── document_ingestion.json
├── tests/
│   ├── conftest.py
│   └── test_api.py
├── data/
│   ├── uploads/                # Uploaded documents
│   └── chroma/                 # ChromaDB persistence
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # Full stack orchestration
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Tool configuration
├── .env.example                # Environment variable template
├── .gitignore
└── README.md
```

---

## 🛠 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.11** | Core backend language |
| **FastAPI** | Async web framework with auto-generated docs |
| **HuggingFace Inference API** | FREE access to open-source AI models |
| **Mistral-7B-Instruct** | Chat / text generation (open-source LLM) |
| **Whisper-large-v3** | Speech-to-text transcription |
| **BLIP** | Image-to-text captioning |
| **sentence-transformers** | Text embeddings for RAG |
| **LangChain** | RAG pipeline orchestration |
| **ChromaDB** | Vector database for embeddings storage |
| **WhatsApp Cloud API** | Business messaging platform integration |
| **n8n** | Visual workflow automation |
| **Docker** | Containerization & deployment |
| **Pydantic** | Data validation & settings management |
| **structlog** | Structured JSON logging |
| **httpx** | Async HTTP client |

---

## ⚙️ Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_API_TOKEN` | ✅ | Your free HuggingFace API token |
| `HF_CHAT_MODEL` | | Chat model (default: `mistralai/Mistral-7B-Instruct-v0.3`) |
| `HF_EMBEDDING_MODEL` | | Embeddings model (default: `sentence-transformers/all-MiniLM-L6-v2`) |
| `HF_WHISPER_MODEL` | | ASR model (default: `openai/whisper-large-v3`) |
| `HF_VISION_MODEL` | | Vision model (default: `Salesforce/blip-image-captioning-large`) |
| `WHATSAPP_API_TOKEN` | For WhatsApp | Meta API access token |
| `WHATSAPP_PHONE_NUMBER_ID` | For WhatsApp | Business phone number ID |
| `WHATSAPP_VERIFY_TOKEN` | For WhatsApp | Webhook verification token |
| `CHROMA_HOST` | | ChromaDB host (default: `chromadb`) |
| `RAG_CHUNK_SIZE` | | Document chunk size (default: `1000`) |
| `RAG_TOP_K` | | Number of RAG results (default: `5`) |

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=html
```

---

## 📱 WhatsApp Setup Guide

1. **Create a Meta Business Account** at [business.facebook.com](https://business.facebook.com)
2. **Create a WhatsApp Business App** in [Meta Developers](https://developers.facebook.com)
3. **Get API Credentials**:
   - Navigate to WhatsApp → API Setup
   - Copy the **Temporary Access Token**
   - Note the **Phone Number ID**
4. **Configure Webhook**:
   - Start ngrok: `ngrok http 8080`
   - Set webhook URL: `https://YOUR_NGROK_URL/api/v1/webhook/whatsapp`
   - Set verify token to match your `WHATSAPP_VERIFY_TOKEN`
   - Subscribe to `messages` field
5. **Update `.env`** with all WhatsApp credentials
6. **Send a test message** to your WhatsApp test number

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

