# Medical Report Processing API

FastAPI server for processing medical documents and generating simplified versions.

## Setup

1. Install dependencies:
```bash
cd api
pip install -r requirements.txt
# Or if using uv:
uv pip install -e .
```

2. Set up environment variables:
Create a `.env` file in the `api` directory with:
```
GEMINI_API_KEY=your_api_key_here
```

3. Run the server:
```bash
python run_server.py
# Or:
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST `/api/process-document`
Processes an uploaded medical document (PDF or TXT) and returns:
- Original text from the document
- Simplified text (plain language version)
- Summary (extracted main ideas)

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (PDF or TXT)

**Response:**
```json
{
  "originalText": "Full original medical report text...",
  "simplifiedText": "Simplified version in plain language...",
  "summary": "Main ideas extracted from the report...",
  "fileName": "document.pdf"
}
```

### GET `/api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Frontend Integration

The React frontend should be configured to call the API at `http://localhost:8000` by default, or set the `VITE_API_URL` environment variable in the client directory.
