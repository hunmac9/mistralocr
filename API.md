# MistraLOCR API Documentation

MistraLOCR provides a RESTful JSON API for programmatic OCR processing of PDF documents.

## Base URL

```
/api/v1
```

## Authentication

The API supports optional API key authentication via the `X-API-Key` header. When provided, rate limits are tracked per API key instead of per IP address.

```bash
curl -H "X-API-Key: your-api-key" https://your-server/api/v1/health
```

## Rate Limits

| Endpoint | Default Limit |
|----------|---------------|
| All endpoints | 100 requests/hour |
| OCR submission | 10 requests/minute |

Rate limits are tracked per IP address (or per API key if provided). When a rate limit is exceeded, the API returns a `429 Too Many Requests` response.

Configure rate limits via environment variables:
- `RATE_LIMIT_DEFAULT`: Default rate limit (e.g., "100 per hour")
- `RATE_LIMIT_OCR`: OCR endpoint rate limit (e.g., "10 per minute")
- `RATE_LIMIT_STORAGE`: Storage backend (e.g., "memory://", "redis://localhost:6379")

---

## Endpoints

### Health Check

Check API and service health status.

```
GET /api/v1/health
```

**Response**

```json
{
  "status": "healthy",
  "service": "mistralocr",
  "version": "1.0.0",
  "components": {
    "api": {"status": "healthy"},
    "local_ocr": {"status": "healthy"},
    "mistral_ocr": {"status": "configured"}
  }
}
```

**Status Codes**
- `200`: Service is healthy
- `503`: Service is degraded

---

### List Backends

Get available OCR backends and their status.

```
GET /api/v1/backends
```

**Response**

```json
{
  "backends": [
    {
      "id": "local",
      "name": "Local OCR (Surya)",
      "status": "available",
      "description": "On-device OCR using Surya (~300M params)"
    },
    {
      "id": "mistral",
      "name": "Mistral OCR API",
      "status": "configured",
      "description": "Cloud-based OCR using Mistral AI API"
    },
    {
      "id": "auto",
      "name": "Automatic",
      "status": "available",
      "description": "Automatically select best available backend"
    }
  ],
  "default": "auto"
}
```

---

### Submit OCR Job

Submit a PDF file for OCR processing.

```
POST /api/v1/ocr
Content-Type: multipart/form-data
```

**Form Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | PDF file to process |
| `backend` | String | No | OCR backend: `auto`, `local`, or `mistral` (default: `auto`) |
| `api_key` | String | No | Mistral API key (can also use header or env) |

**Example Request**

```bash
curl -X POST \
  -F "file=@document.pdf" \
  -F "backend=local" \
  https://your-server/api/v1/ocr
```

**Response (202 Accepted)**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Job submitted successfully",
  "backend": "local",
  "filename": "document.pdf",
  "links": {
    "status": "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000",
    "stream": "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/stream"
  }
}
```

**Error Responses**

| Status | Code | Description |
|--------|------|-------------|
| `400` | `MISSING_FILE` | No file provided |
| `400` | `EMPTY_FILENAME` | Empty filename |
| `400` | `INVALID_FILE_TYPE` | File type not allowed |
| `400` | `INVALID_BACKEND` | Invalid backend specified |
| `400` | `BACKEND_ERROR` | Backend initialization failed |
| `413` | `FILE_TOO_LARGE` | File exceeds size limit |
| `415` | `INVALID_CONTENT_TYPE` | Invalid content type |
| `429` | `RATE_LIMIT_EXCEEDED` | Rate limit exceeded |

---

### Get Job Status

Get the current status of an OCR job.

```
GET /api/v1/jobs/{job_id}
```

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | UUID | The job identifier |

**Response**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "done",
  "backend": "local",
  "filename": "document.pdf",
  "created_at": "2024-01-15T10:30:00Z",
  "download_url": "https://your-server/download/session-id/document_output.zip"
}
```

**Job Status Values**

| Status | Description |
|--------|-------------|
| `queued` | Job is waiting to be processed |
| `processing` | OCR is in progress |
| `done` | Processing complete, result available |
| `error` | Processing failed |

**Error Response (status=error)**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "error",
  "backend": "local",
  "filename": "document.pdf",
  "created_at": "2024-01-15T10:30:00Z",
  "error": "OCR processing failed: invalid PDF format"
}
```

---

### Stream Job Status (SSE)

Stream real-time job status updates via Server-Sent Events.

```
GET /api/v1/jobs/{job_id}/stream
```

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | UUID | The job identifier |

**Example Client (JavaScript)**

```javascript
const eventSource = new EventSource('/api/v1/jobs/550e8400.../stream');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Status:', data.status);

  if (data.status === 'done') {
    console.log('Download URL:', data.download_url);
    eventSource.close();
  } else if (data.status === 'error') {
    console.error('Error:', data.error);
    eventSource.close();
  }
};
```

**Example Client (curl)**

```bash
curl -N https://your-server/api/v1/jobs/550e8400.../stream
```

---

### Download Job Result

Download the result of a completed OCR job.

```
GET /api/v1/jobs/{job_id}/result
```

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | UUID | The job identifier |

**Response**

- `200`: ZIP file download
- `404`: Job not found or not complete
- `410`: Result expired/cleaned up

**ZIP Contents**

```
document_output.zip
├── document.md      # Extracted text in Markdown format
├── img-1.webp       # Extracted images (WebP format)
├── img-2.webp
└── ...
```

---

## Error Response Format

All error responses follow a consistent format:

```json
{
  "error": {
    "message": "Human-readable error message",
    "status": 400,
    "code": "ERROR_CODE"
  }
}
```

**Common Error Codes**

| Code | Status | Description |
|------|--------|-------------|
| `MISSING_FILE` | 400 | No file was provided |
| `INVALID_FILE_TYPE` | 400 | File type not supported |
| `INVALID_BACKEND` | 400 | Invalid backend specified |
| `JOB_NOT_FOUND` | 404 | Job ID does not exist |
| `JOB_NOT_COMPLETE` | 404 | Job is still processing |
| `RESULT_EXPIRED` | 410 | Result files cleaned up |
| `FILE_TOO_LARGE` | 413 | File exceeds size limit |
| `INVALID_CONTENT_TYPE` | 415 | Wrong content type |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Complete Example

Here's a complete workflow using the API:

```bash
# 1. Check service health
curl https://your-server/api/v1/health

# 2. Submit a PDF for OCR
JOB_RESPONSE=$(curl -s -X POST \
  -F "file=@document.pdf" \
  https://your-server/api/v1/ocr)

JOB_ID=$(echo $JOB_RESPONSE | jq -r '.job_id')
echo "Job ID: $JOB_ID"

# 3. Poll for job status
while true; do
  STATUS=$(curl -s https://your-server/api/v1/jobs/$JOB_ID | jq -r '.status')
  echo "Status: $STATUS"

  if [ "$STATUS" = "done" ]; then
    break
  elif [ "$STATUS" = "error" ]; then
    echo "Error occurred!"
    exit 1
  fi

  sleep 2
done

# 4. Download the result
curl -o result.zip https://your-server/api/v1/jobs/$JOB_ID/result

# 5. Extract and view
unzip result.zip
cat *.md
```

---

## Python Client Example

```python
import requests
import time

BASE_URL = "https://your-server/api/v1"

def ocr_pdf(file_path: str, backend: str = "auto") -> dict:
    """Submit a PDF for OCR and wait for results."""

    # Submit job
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/ocr",
            files={"file": f},
            data={"backend": backend}
        )
    response.raise_for_status()
    job = response.json()
    job_id = job["job_id"]
    print(f"Job submitted: {job_id}")

    # Poll for completion
    while True:
        response = requests.get(f"{BASE_URL}/jobs/{job_id}")
        response.raise_for_status()
        status = response.json()

        if status["status"] == "done":
            return status
        elif status["status"] == "error":
            raise Exception(status.get("error", "Unknown error"))

        time.sleep(2)

def download_result(job_id: str, output_path: str):
    """Download the result ZIP file."""
    response = requests.get(f"{BASE_URL}/jobs/{job_id}/result", stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# Usage
result = ocr_pdf("document.pdf", backend="local")
print(f"Download URL: {result['download_url']}")
download_result(result["job_id"], "result.zip")
```

---

## Notes

- **File Size Limits**: Default maximum upload size is 100MB. PDFs larger than 50MB are automatically split into parts for processing.
- **Supported Formats**: Only PDF files are currently supported.
- **Image Output**: Extracted images are converted to WebP format for optimal size/quality.
- **Job Retention**: Job results are stored temporarily. Download results promptly after completion.
- **Concurrency**: Multiple jobs can be processed concurrently. Each job runs in a background thread.
