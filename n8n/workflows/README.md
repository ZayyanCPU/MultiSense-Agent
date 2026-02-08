# n8n Workflows for MultiSense Agent

This directory contains pre-built n8n workflow templates for the MultiSense Agent.

## Workflows

### 1. `whatsapp_chatbot.json`
**Main WhatsApp chatbot workflow.**

Flow:
1. Receives WhatsApp webhook POST
2. Returns 200 OK immediately
3. Parses the message payload
4. Routes by message type (text/voice/image/document)
5. Calls the MultiSense Agent API
6. Sends response back via WhatsApp

### 2. `document_ingestion.json`
**Document processing pipeline.**

Flow:
1. Receives document upload via webhook
2. Validates file type (PDF)
3. Forwards to the RAG engine for ingestion
4. Returns ingestion status

## How to Import

1. Open n8n at `http://localhost:5678`
2. Go to **Workflows** → **Import from File**
3. Select the `.json` file
4. Configure credentials (WhatsApp Business Cloud, HTTP Request)
5. Activate the workflow

## Required Credentials in n8n

- **WhatsApp Business Cloud**: Your WhatsApp API token and phone number ID
- **HTTP Request** (for internal API calls): No auth needed (internal network)
