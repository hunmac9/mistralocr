# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
OCR Backend Abstraction Layer

This module provides a unified interface for different OCR backends:
- Mistral OCR (cloud-based)
- Local OCR using Surya or Chandra (via Docker container)

The local OCR is the default and preferred option as it doesn't require
an API key and keeps data private.
"""

import os
import time
import json
import base64
import subprocess
import threading
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable

import requests

# Type alias for progress callback: (message: str) -> None
ProgressCallback = Optional[Callable[[str], None]]
from PIL import Image
import io


@dataclass
class OCRImage:
    """Represents an extracted image from OCR."""
    id: str
    image_base64: str


@dataclass
class OCRPage:
    """Represents a single page of OCR results."""
    index: int
    markdown: str
    images: list[OCRImage] = field(default_factory=list)


@dataclass
class OCRResponse:
    """Represents the full OCR response."""
    pages: list[OCRPage]
    model: str
    processing_time: float = 0.0


class OCRBackend(ABC):
    """Abstract base class for OCR backends."""

    @abstractmethod
    def process(self, pdf_path: Path, on_progress: ProgressCallback = None) -> OCRResponse:
        """
        Process a PDF file and return OCR results.

        Args:
            pdf_path: Path to the PDF file to process
            on_progress: Optional callback function called with progress messages

        Returns:
            OCRResponse with extracted text and images
        """
        pass

    def _report(self, on_progress: ProgressCallback, message: str):
        """Helper to safely call progress callback."""
        if on_progress:
            try:
                on_progress(message)
            except Exception:
                pass  # Don't let callback errors break processing

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available for use."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the display name of this backend."""
        pass


class MistralOCRBackend(OCRBackend):
    """OCR backend using Mistral AI API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy initialize the Mistral client."""
        if self._client is None:
            from mistralai import Mistral
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_name(self) -> str:
        return "Mistral OCR"

    def process(self, pdf_path: Path, on_progress: ProgressCallback = None) -> OCRResponse:
        from mistralai import DocumentURLChunk

        client = self._get_client()
        uploaded_file_id = None

        try:
            self._report(on_progress, "Uploading to Mistral")
            print(f"  [Mistral] Uploading {pdf_path.name}...")
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            mistral_file = client.files.upload(
                file={"file_name": pdf_path.name, "content": pdf_bytes},
                purpose="ocr"
            )
            uploaded_file_id = mistral_file.id

            self._report(on_progress, "Preparing document")
            print(f"  [Mistral] File uploaded (ID: {uploaded_file_id}). Getting signed URL...")
            signed_url = client.files.get_signed_url(file_id=uploaded_file_id, expiry=300)

            self._report(on_progress, "Running Mistral OCR")
            print(f"  [Mistral] Calling OCR API...")
            start_time = time.time()
            ocr_result = client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
            processing_time = time.time() - start_time
            print(f"  [Mistral] OCR complete in {processing_time:.2f}s")
            self._report(on_progress, f"OCR complete ({processing_time:.1f}s)")

            # Convert to our standard format
            pages = []
            for page_idx, page in enumerate(ocr_result.pages):
                images = []
                for img in page.images:
                    if img.id and img.image_base64:
                        images.append(OCRImage(id=img.id, image_base64=img.image_base64))

                pages.append(OCRPage(
                    index=page_idx,
                    markdown=page.markdown,
                    images=images
                ))

            return OCRResponse(
                pages=pages,
                model="mistral-ocr-latest",
                processing_time=processing_time
            )

        finally:
            # Clean up uploaded file
            if uploaded_file_id:
                try:
                    print(f"  [Mistral] Cleaning up temporary file...")
                    client.files.delete(file_id=uploaded_file_id)
                except Exception as e:
                    print(f"  [Mistral] Warning: Could not delete file: {e}")



class LocalOCRBackend(OCRBackend):
    """
    OCR backend using local models via Docker container.

    Supports multiple models:
    - Surya OCR (surya): CPU-friendly, fast (~300M params)
    - Chandra OCR (chandra): GPU required, best quality (9B params)

    This backend manages the Docker container lifecycle automatically:
    - Starts the container on first use
    - Keeps it running while processing
    - Stops after idle timeout (configurable)
    - Switches between models as requested
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        container_name: str = "mistralocr-local-ocr",
        docker_image: str = "mistralocr-local-ocr:latest",
        idle_timeout: int = 300,
        auto_start: bool = True,
        use_docker: bool = True,
        local_model: str = "surya",  # "surya" or "chandra"
    ):
        self.server_url = server_url.rstrip('/')
        self.container_name = container_name
        self.docker_image = docker_image
        self.idle_timeout = idle_timeout
        self.auto_start = auto_start
        self.use_docker = use_docker
        self.local_model = local_model
        self._container_started = False
        self._start_lock = threading.Lock()

    def is_available(self) -> bool:
        """Check if the local OCR server is reachable."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_name(self) -> str:
        if self.local_model == "chandra":
            return "Local OCR (Chandra)"
        return "Local OCR (Surya)"

    def _is_container_running(self) -> bool:
        """Check if the Docker container is running."""
        if not self.use_docker:
            return True

        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip() == "true"
        except:
            return False

    def _start_container(self) -> bool:
        """Start the Docker container if not running."""
        if not self.use_docker:
            return True

        with self._start_lock:
            if self._is_container_running():
                print(f"  [Local OCR] Container {self.container_name} is already running")
                return True

            print(f"  [Local OCR] Starting container {self.container_name}...")

            # Remove existing container if exists (stopped state)
            try:
                subprocess.run(
                    ["docker", "rm", "-f", self.container_name],
                    capture_output=True,
                    timeout=30
                )
            except:
                pass

            # Start new container
            try:
                # Get port from server URL
                port = 8000
                if ":" in self.server_url.replace("http://", "").replace("https://", ""):
                    port = int(self.server_url.split(":")[-1].split("/")[0])

                cmd = [
                    "docker", "run", "-d",
                    "--name", self.container_name,
                    "-p", f"{port}:{port}",
                    "-e", f"IDLE_TIMEOUT={self.idle_timeout}",
                    "-e", f"PORT={port}",
                    # Mount volume for model cache to persist downloads
                    "-v", "mistralocr-cache:/home/appuser/.cache",
                    self.docker_image
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                if result.returncode != 0:
                    print(f"  [Local OCR] Failed to start container: {result.stderr}")
                    return False

                # Wait for container to be healthy
                print(f"  [Local OCR] Waiting for container to be ready...")
                for i in range(60):  # Wait up to 60 seconds
                    time.sleep(1)
                    if self.is_available():
                        print(f"  [Local OCR] Container ready after {i+1} seconds")
                        self._container_started = True
                        return True

                print(f"  [Local OCR] Container failed to become healthy")
                return False

            except subprocess.TimeoutExpired:
                print(f"  [Local OCR] Timeout starting container")
                return False
            except Exception as e:
                print(f"  [Local OCR] Error starting container: {e}")
                return False

    def _stop_container(self):
        """Stop the Docker container."""
        if not self.use_docker:
            return

        try:
            subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True,
                timeout=30
            )
            print(f"  [Local OCR] Container {self.container_name} stopped")
        except Exception as e:
            print(f"  [Local OCR] Error stopping container: {e}")

    def process(self, pdf_path: Path, on_progress: ProgressCallback = None) -> OCRResponse:
        """Process a PDF using the local OCR server with streaming progress."""

        # Ensure server is running
        if not self.is_available():
            if self.auto_start:
                self._report(on_progress, "Starting OCR server")
                if not self._start_container():
                    raise RuntimeError("Failed to start local OCR container")
                # Wait a bit more for model to potentially load
                if not self.is_available():
                    raise RuntimeError("Local OCR server not responding after start")
            else:
                raise RuntimeError("Local OCR server is not available")

        model_display = "Chandra" if self.local_model == "chandra" else "Surya"
        self._report(on_progress, f"Sending to {model_display}")
        print(f"  [Local OCR] Processing {pdf_path.name} with {model_display}...")
        start_time = time.time()

        try:
            # Use streaming endpoint for progress updates
            with open(pdf_path, 'rb') as f:
                files = {'file': (pdf_path.name, f, 'application/pdf')}
                data = {'task': 'ocr', 'include_images': 'true', 'model': self.local_model}

                response = requests.post(
                    f"{self.server_url}/ocr/stream",
                    files=files,
                    data=data,
                    timeout=600,  # 10 minute timeout for large files
                    stream=True  # Enable streaming
                )

            if response.status_code != 200:
                try:
                    error_msg = response.json().get('error', 'Unknown error')
                except:
                    error_msg = f"HTTP {response.status_code}"
                raise RuntimeError(f"OCR server error: {error_msg}")

            # Process streaming response (NDJSON)
            result = None
            for line in response.iter_lines():
                if line:
                    try:
                        msg = json.loads(line.decode('utf-8'))
                        msg_type = msg.get('type')

                        if msg_type == 'status':
                            self._report(on_progress, msg.get('message', ''))
                        elif msg_type == 'progress':
                            page = msg.get('page', 0)
                            total = msg.get('total', 0)
                            self._report(on_progress, f"Page {page}/{total}")
                        elif msg_type == 'result':
                            result = msg
                        elif msg_type == 'error':
                            raise RuntimeError(msg.get('message', 'OCR error'))
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

            if result is None:
                raise RuntimeError("No result received from OCR server")

            processing_time = time.time() - start_time
            print(f"  [Local OCR] OCR complete in {processing_time:.2f}s")

            # Report completion
            page_count = len(result.get('pages', []))
            self._report(on_progress, f"Completed {page_count} pages")

            # Convert to our standard format
            pages = []
            for page_data in result.get('pages', []):
                images = []
                for img in page_data.get('images', []):
                    images.append(OCRImage(
                        id=img.get('id', ''),
                        image_base64=img.get('image_base64', '')
                    ))

                pages.append(OCRPage(
                    index=page_data.get('index', 0),
                    markdown=page_data.get('markdown', ''),
                    images=images
                ))

            return OCRResponse(
                pages=pages,
                model=result.get('model', 'Surya'),
                processing_time=processing_time
            )

        except requests.exceptions.Timeout:
            raise RuntimeError("OCR request timed out")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Failed to connect to local OCR server")


def get_ocr_backend(
    backend_type: str = "auto",
    mistral_api_key: Optional[str] = None,
    local_server_url: str = "http://localhost:8000",
    local_model: str = "surya",
    **kwargs
) -> OCRBackend:
    """
    Factory function to get the appropriate OCR backend.

    Args:
        backend_type: One of "auto", "local", or "mistral"
        mistral_api_key: API key for Mistral (required if backend_type is "mistral")
        local_server_url: URL of local OCR server
        local_model: Local model to use ("surya" or "chandra")
        **kwargs: Additional arguments passed to backend constructor

    Returns:
        An OCRBackend instance

    Backend selection logic for "auto":
    1. Try local OCR first (default)
    2. Fall back to Mistral if API key is provided and local is unavailable
    """
    if backend_type == "mistral":
        if not mistral_api_key:
            raise ValueError("Mistral API key required for Mistral backend")
        return MistralOCRBackend(api_key=mistral_api_key)

    elif backend_type == "local":
        return LocalOCRBackend(server_url=local_server_url, local_model=local_model, **kwargs)

    else:  # auto
        # Try local first (default)
        local_backend = LocalOCRBackend(server_url=local_server_url, local_model=local_model, **kwargs)

        # Check if local is available or can be started
        if local_backend.is_available():
            model_name = "Chandra" if local_model == "chandra" else "Surya"
            print(f"Using local OCR backend with {model_name} (already running)")
            return local_backend

        # If auto_start is enabled, try to start it
        if kwargs.get('auto_start', True):
            print("Local OCR not running, will start on first use")
            return local_backend

        # Fall back to Mistral if API key is available
        if mistral_api_key:
            print("Local OCR not available, falling back to Mistral OCR")
            return MistralOCRBackend(api_key=mistral_api_key)

        # Default to local with auto_start
        print("Using local OCR backend (will start container on first use)")
        return local_backend
