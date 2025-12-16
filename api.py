# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
MistraLOCR API Module

This module provides a clean, documented JSON API for programmatic access
to OCR functionality. All endpoints return JSON responses with consistent
error handling and rate limiting.
"""

import os
import queue
import threading
from pathlib import Path
from uuid import uuid4
from functools import wraps

from flask import Blueprint, request, jsonify, Response, current_app
from werkzeug.utils import secure_filename

# API Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')


def get_api_key_or_ip():
    """Get identifier for rate limiting - API key if provided, else IP address."""
    api_key = request.headers.get('X-API-Key')
    if api_key:
        return f"key:{api_key}"
    return f"ip:{request.remote_addr}"


def json_response(data, status=200):
    """Create a consistent JSON response."""
    response = jsonify(data)
    response.status_code = status
    return response


def error_response(message, status=400, code=None):
    """Create a consistent error response."""
    error_data = {"error": {"message": message, "status": status}}
    if code:
        error_data["error"]["code"] = code
    return json_response(error_data, status)


def require_json_or_form(f):
    """Decorator to ensure request has proper content type for POST requests."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method == 'POST':
            content_type = request.content_type or ''
            if not any(ct in content_type for ct in ['application/json', 'multipart/form-data', 'application/x-www-form-urlencoded']):
                return error_response(
                    "Content-Type must be application/json or multipart/form-data",
                    status=415,
                    code="INVALID_CONTENT_TYPE"
                )
        return f(*args, **kwargs)
    return decorated


# ============================================================================
# API Endpoints
# ============================================================================

@api_bp.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.

    Returns the health status of the API and its dependencies.

    Response:
        200: Service is healthy
        503: Service is degraded or unhealthy
    """
    import requests as http_requests

    # Get configuration from app context
    local_ocr_url = current_app.config.get('LOCAL_OCR_URL', 'http://localhost:8000')

    health_status = {
        "status": "healthy",
        "service": "mistralocr",
        "version": "1.0.0",
        "components": {
            "api": {"status": "healthy"},
            "local_ocr": {"status": "unknown"},
            "mistral_ocr": {"status": "unknown"}
        }
    }

    # Check local OCR
    try:
        resp = http_requests.get(f"{local_ocr_url}/health", timeout=2)
        if resp.status_code == 200:
            health_status["components"]["local_ocr"]["status"] = "healthy"
        else:
            health_status["components"]["local_ocr"]["status"] = "degraded"
    except Exception:
        health_status["components"]["local_ocr"]["status"] = "unavailable"

    # Check Mistral API key availability
    if os.getenv("MISTRAL_API_KEY"):
        health_status["components"]["mistral_ocr"]["status"] = "configured"
    else:
        health_status["components"]["mistral_ocr"]["status"] = "not_configured"

    # Overall health
    if all(c["status"] in ["healthy", "configured", "not_configured"]
           for c in health_status["components"].values()):
        return json_response(health_status, 200)
    else:
        health_status["status"] = "degraded"
        return json_response(health_status, 503)


@api_bp.route('/ocr', methods=['POST'])
@require_json_or_form
def submit_ocr_job():
    """
    Submit a PDF for OCR processing.

    Request:
        Content-Type: multipart/form-data

        Form fields:
            - file: PDF file to process (required)
            - backend: OCR backend to use - "auto", "local", or "mistral" (optional, default: "auto")
            - api_key: Mistral API key (optional, can also use X-API-Key header or MISTRAL_API_KEY env)
            - webhook_url: URL to POST results to when complete (optional)

    Response:
        202: Job accepted for processing
            {
                "job_id": "uuid",
                "status": "queued",
                "message": "Job submitted successfully",
                "links": {
                    "status": "/api/v1/jobs/{job_id}",
                    "stream": "/api/v1/jobs/{job_id}/stream"
                }
            }

        400: Bad request (missing file, invalid format)
        413: File too large
        415: Invalid content type
        429: Rate limit exceeded
        500: Server error
    """
    # Import here to avoid circular imports
    from ocr_backends import get_ocr_backend

    # Get app configuration
    upload_folder = Path(current_app.config.get('UPLOAD_FOLDER', 'uploads'))
    ocr_backend_default = current_app.config.get('OCR_BACKEND', 'auto')
    local_ocr_url = current_app.config.get('LOCAL_OCR_URL', 'http://localhost:8000')
    local_ocr_container = current_app.config.get('LOCAL_OCR_CONTAINER_NAME', 'mistralocr-paddleocr')
    local_ocr_image = current_app.config.get('LOCAL_OCR_DOCKER_IMAGE', 'mistralocr-paddleocr:latest')
    local_ocr_timeout = current_app.config.get('LOCAL_OCR_IDLE_TIMEOUT', 300)
    local_ocr_auto_start = current_app.config.get('LOCAL_OCR_AUTO_START', True)
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'pdf'})

    # Get shared state from app
    jobs = current_app.config['jobs']
    job_queues = current_app.config['job_queues']
    jobs_lock = current_app.config['jobs_lock']
    background_process_job = current_app.config['background_process_job']

    # Check for file
    if 'file' not in request.files:
        return error_response("No file provided", status=400, code="MISSING_FILE")

    file = request.files['file']
    if file.filename == '':
        return error_response("No file selected", status=400, code="EMPTY_FILENAME")

    # Validate file extension
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return error_response(
            f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}",
            status=400,
            code="INVALID_FILE_TYPE"
        )

    # Get backend configuration
    backend_type = request.form.get('backend', ocr_backend_default)
    if backend_type not in ['auto', 'local', 'mistral']:
        return error_response(
            "Invalid backend. Must be 'auto', 'local', or 'mistral'",
            status=400,
            code="INVALID_BACKEND"
        )

    # Get API key from various sources (priority: form > header > env)
    api_key = (
        request.form.get('api_key') or
        request.headers.get('X-API-Key') or
        os.getenv("MISTRAL_API_KEY")
    )

    # Create OCR backend
    try:
        ocr_backend = get_ocr_backend(
            backend_type=backend_type,
            mistral_api_key=api_key,
            local_server_url=local_ocr_url,
            container_name=local_ocr_container,
            docker_image=local_ocr_image,
            idle_timeout=local_ocr_timeout,
            auto_start=local_ocr_auto_start,
        )
    except ValueError as e:
        return error_response(str(e), status=400, code="BACKEND_ERROR")

    # Generate IDs
    job_id = str(uuid4())
    session_id = str(uuid4())

    # Save file
    try:
        session_upload_dir = upload_folder / session_id
        session_upload_dir.mkdir(parents=True, exist_ok=True)
        filename_sanitized = secure_filename(file.filename)
        temp_pdf_path = session_upload_dir / filename_sanitized
        file.save(temp_pdf_path)
    except Exception as e:
        return error_response(f"Failed to save file: {e}", status=500, code="SAVE_ERROR")

    # Initialize job
    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "download_url": None,
            "error": None,
            "session_id": session_id,
            "backend": ocr_backend.get_name(),
            "filename": filename_sanitized,
            "created_at": __import__('datetime').datetime.utcnow().isoformat() + "Z"
        }
        job_queues[job_id] = queue.Queue()

    # Start background processing
    t = threading.Thread(
        target=background_process_job,
        args=(job_id, temp_pdf_path, ocr_backend, session_id)
    )
    t.daemon = True
    t.start()

    return json_response({
        "job_id": job_id,
        "status": "queued",
        "message": "Job submitted successfully",
        "backend": ocr_backend.get_name(),
        "filename": filename_sanitized,
        "links": {
            "status": f"/api/v1/jobs/{job_id}",
            "stream": f"/api/v1/jobs/{job_id}/stream"
        }
    }, status=202)


@api_bp.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """
    Get the status of an OCR job.

    Parameters:
        job_id: The unique job identifier

    Response:
        200: Job found
            {
                "job_id": "uuid",
                "status": "queued|processing|done|error",
                "backend": "local|mistral",
                "filename": "document.pdf",
                "created_at": "2024-01-01T00:00:00Z",
                "download_url": "url" (only when status=done),
                "error": "message" (only when status=error)
            }

        404: Job not found
        429: Rate limit exceeded
    """
    jobs = current_app.config['jobs']
    jobs_lock = current_app.config['jobs_lock']

    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return error_response("Job not found", status=404, code="JOB_NOT_FOUND")

    response_data = {
        "job_id": job_id,
        "status": job["status"],
        "backend": job.get("backend"),
        "filename": job.get("filename"),
        "created_at": job.get("created_at")
    }

    if job["status"] == "done" and job.get("download_url"):
        response_data["download_url"] = job["download_url"]
    elif job["status"] == "error" and job.get("error"):
        response_data["error"] = job["error"]

    return json_response(response_data)


@api_bp.route('/jobs/<job_id>/stream', methods=['GET'])
def stream_job_status(job_id):
    """
    Stream job status updates via Server-Sent Events (SSE).

    Parameters:
        job_id: The unique job identifier

    Response:
        200: SSE stream
            Each event is a JSON object with job status:
            data: {"status": "processing"}
            data: {"status": "done", "download_url": "..."}

        404: Job not found
        429: Rate limit exceeded

    Notes:
        - Connection will be held open until job completes or times out (310s)
        - Client should reconnect if disconnected before completion
    """
    import json as json_module

    jobs = current_app.config['jobs']
    job_queues = current_app.config['job_queues']
    jobs_lock = current_app.config['jobs_lock']

    def event_stream():
        q = job_queues.get(job_id)
        if not q:
            with jobs_lock:
                job = jobs.get(job_id)
                if job:
                    yield f"data: {json_module.dumps(job)}\n\n"
                else:
                    yield f"data: {json_module.dumps({'status': 'error', 'error': 'Job not found'})}\n\n"
            return

        sse_timeout = 310
        try:
            while True:
                message = q.get(timeout=sse_timeout)
                if message is None:
                    break
                yield f"data: {message}\n\n"
        except queue.Empty:
            pass
        finally:
            with jobs_lock:
                if job_id in job_queues:
                    del job_queues[job_id]

    return Response(event_stream(), mimetype="text/event-stream")


@api_bp.route('/jobs/<job_id>/result', methods=['GET'])
def get_job_result(job_id):
    """
    Download the result of a completed OCR job.

    Parameters:
        job_id: The unique job identifier

    Response:
        200: ZIP file download
        404: Job not found or not complete
        410: Result expired/cleaned up
        429: Rate limit exceeded

    Notes:
        - Results are available as ZIP archives containing:
          - Markdown file with extracted text
          - WebP images extracted from the PDF
    """
    from flask import send_from_directory

    jobs = current_app.config['jobs']
    jobs_lock = current_app.config['jobs_lock']
    output_folder = Path(current_app.config.get('OUTPUT_FOLDER', 'output'))

    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return error_response("Job not found", status=404, code="JOB_NOT_FOUND")

    if job["status"] != "done":
        return error_response(
            f"Job not complete. Current status: {job['status']}",
            status=404,
            code="JOB_NOT_COMPLETE"
        )

    session_id = job.get("session_id")
    filename = job.get("filename", "").replace('.pdf', '_output.zip')
    filename = secure_filename(filename.rsplit('.', 1)[0]) + "_output.zip"

    directory = output_folder / session_id
    file_path = directory / filename

    if not file_path.exists():
        return error_response(
            "Result file not found. It may have been cleaned up.",
            status=410,
            code="RESULT_EXPIRED"
        )

    return send_from_directory(directory, filename, as_attachment=True)


@api_bp.route('/backends', methods=['GET'])
def list_backends():
    """
    List available OCR backends and their status.

    Response:
        200: List of available backends
            {
                "backends": [
                    {
                        "id": "local",
                        "name": "Local PaddleOCR-VL",
                        "status": "available|unavailable",
                        "description": "On-device OCR using PaddleOCR-VL model"
                    },
                    {
                        "id": "mistral",
                        "name": "Mistral OCR API",
                        "status": "configured|not_configured",
                        "description": "Cloud-based OCR using Mistral AI"
                    }
                ],
                "default": "auto"
            }
    """
    import requests as http_requests

    local_ocr_url = current_app.config.get('LOCAL_OCR_URL', 'http://localhost:8000')
    default_backend = current_app.config.get('OCR_BACKEND', 'auto')

    # Check local OCR availability
    local_status = "unavailable"
    try:
        resp = http_requests.get(f"{local_ocr_url}/health", timeout=2)
        if resp.status_code == 200:
            local_status = "available"
    except Exception:
        pass

    # Check Mistral configuration
    mistral_status = "configured" if os.getenv("MISTRAL_API_KEY") else "not_configured"

    return json_response({
        "backends": [
            {
                "id": "local",
                "name": "Local PaddleOCR-VL",
                "status": local_status,
                "description": "On-device OCR using PaddleOCR-VL model (0.9B parameters)"
            },
            {
                "id": "mistral",
                "name": "Mistral OCR API",
                "status": mistral_status,
                "description": "Cloud-based OCR using Mistral AI API"
            },
            {
                "id": "auto",
                "name": "Automatic",
                "status": "available",
                "description": "Automatically select best available backend"
            }
        ],
        "default": default_backend
    })


# ============================================================================
# Error Handlers for API Blueprint
# ============================================================================

@api_bp.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    max_mb = current_app.config.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024) // (1024 * 1024)
    return error_response(
        f"File too large. Maximum size is {max_mb}MB",
        status=413,
        code="FILE_TOO_LARGE"
    )


@api_bp.errorhandler(429)
def rate_limit_exceeded(error):
    """Handle rate limit errors."""
    return error_response(
        "Rate limit exceeded. Please slow down your requests.",
        status=429,
        code="RATE_LIMIT_EXCEEDED"
    )


@api_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    return error_response(
        "An internal error occurred",
        status=500,
        code="INTERNAL_ERROR"
    )
