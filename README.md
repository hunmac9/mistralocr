# PDF-to-Markdown OCR Web App

[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-WebApp-green?logo=flask)](https://flask.palletsprojects.com/)

A web application using Flask to convert PDF files into standard Markdown documents, extracting text and images. Supports both **local OCR** (using Surya) and **cloud OCR** (using Mistral AI API).

*Inspired by an Obsidian version at [diegomarzaa/pdf-ocr-obsidian](https://github.com/diegomarzaa/pdf-ocr-obsidian)*

*This version is specifically adapted to import to a modified version of Outline, but should work with most markdown importers*

## Features

-   **Local OCR (Default):** Uses Surya OCR (~300M params, CPU and GPU friendly) for privacy-preserving, no-API-key-required OCR
-   **Cloud OCR (Optional):** Leverages [Mistral OCR](https://mistral.ai/news/mistral-ocr) for accurate text/image extraction
-   **Upload Interface:** Simple web UI for uploading PDFs with OCR engine selection
-   **Standard Markdown:** Outputs clean Markdown with relative image links
-   **Image Handling:** Saves extracted images alongside the Markdown file in WebP format
-   **Packaged Output:** Delivers results (Markdown + images) as a downloadable ZIP archive
-   **Memory Efficient:** Local OCR model automatically unloads after idle timeout to save memory
-   **Configurable Upload Limit:** Set maximum PDF upload size via environment variable (default 100MB)
-   **Dockerized:** Ready to run with Docker Compose

## Getting Started

### Prerequisites

-   [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/) installed
-   **CPU Mode:** 8GB+ RAM recommended (uses ~4-6GB when model is loaded)
-   **GPU Mode (Optional):** NVIDIA GPU with 6GB+ VRAM, CUDA support, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
-   (Optional) A Mistral AI API Key from [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys) if you want to use cloud OCR

### Running the App

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/hunmac9/mistralocr.git
    cd mistralocr
    ```

2. **Configure Environment Variables:**

    - Copy the example environment file:

    ```bash
    cp .env.example .env
    ```

    - Optionally customize settings in `.env` (see Configuration section below)

3. **Launch with Docker Compose:**

    **CPU Mode (default):**
    ```bash
    docker compose up --build -d
    ```

    **GPU Mode (NVIDIA):**
    ```bash
    docker compose -f docker-compose.gpu.yml up --build -d
    ```

    * `--build`: Rebuilds the images if needed
    * `-d`: Runs the containers in the background (detached mode)

    **Note:** The first run will download the Surya OCR model (~1GB), which may take a few minutes.

4. **Access the Web App:**

    Open your browser and go to:

    ```
    http://localhost:5009
    ```

    (Or replace `5009` with your configured `FLASK_PORT`)

5. **Stopping the App:**

    ```bash
    docker compose down
    ```

## Configuration

All configuration is done via environment variables in the `.env` file:

### OCR Backend Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_BACKEND` | `auto` | OCR engine: `auto`, `local`, or `mistral` |
| `LOCAL_OCR_IDLE_TIMEOUT` | `300` | Seconds before local OCR model unloads from memory |
| `MISTRAL_API_KEY` | - | Mistral API key (only needed for Mistral backend) |

**Backend Modes:**
- `auto`: Uses local OCR by default, falls back to Mistral if API key is available
- `local`: Uses only local OCR (Surya, no API key required)
- `mistral`: Uses only Mistral OCR API (requires API key)

### Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_PORT` | `5009` | Port the application runs on |
| `SERVER_NAME` | `localhost:5009` | Hostname for generating download URLs |
| `MAX_UPLOAD_MB` | `100` | Maximum upload size in MB |
| `MISTRAL_MAX_MB` | `50` | Max file size before splitting (applies to all backends) |

## How It Works

1. User uploads PDF(s) via the web UI and selects an OCR engine
2. The Flask backend sends the PDF to the selected OCR backend:
   - **Local:** Surya OCR processes the PDF locally in Docker
   - **Cloud:** PDF is uploaded to Mistral OCR API
3. OCR engine returns markdown-formatted text and images
4. The app saves the markdown and images in a folder
5. Everything is zipped up and served to the web UI

### Memory Management

The local OCR model (Surya) uses ~2-4GB RAM on CPU or ~2GB VRAM on GPU.

To minimize resource usage:

- The model is **only loaded when needed** (lazy loading)
- After processing, the model **automatically unloads** after the idle timeout (default: 5 minutes)
- When unloaded, the service uses minimal memory (~100MB)

You can adjust the idle timeout via `LOCAL_OCR_IDLE_TIMEOUT` environment variable.

## GPU Acceleration

GPU mode provides **10-50x faster** OCR processing compared to CPU.

### Requirements

**Linux:**
1. NVIDIA GPU with CUDA support (6GB+ VRAM recommended)
2. NVIDIA drivers installed
3. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

**Windows (Docker Desktop):**
1. NVIDIA GPU with CUDA support
2. NVIDIA drivers installed in Windows
3. Docker Desktop with WSL2 backend enabled
4. GPU support enabled in Docker Desktop settings

### Usage

```bash
# Build and run with GPU support
docker compose -f docker-compose.gpu.yml up --build -d

# Check GPU is detected
docker compose -f docker-compose.gpu.yml logs local-ocr | grep "Using device"
# Should show: Using device: cuda (NVIDIA GeForce RTX XXXX, X.XGB VRAM)
```

### GPU Memory Management

When using GPU mode:
- Model uses ~2-3GB VRAM (bfloat16 precision)
- VRAM is **automatically freed** after the idle timeout
- Logs show VRAM usage before/after loading and unloading
- Falls back to CPU if GPU is not available

## Architecture

```
+-------------------+      +-------------------+
|                   |      |                   |
|   Main App        | ---> |   Local OCR       |
|   (Flask)         |      |   (Surya)         |
|   Port: 5009      |      |   Port: 8000      |
|                   |      |                   |
+-------------------+      +-------------------+
        |
        | (optional, if MISTRAL_API_KEY set)
        v
+-------------------+
|                   |
|   Mistral OCR     |
|   (Cloud API)     |
|                   |
+-------------------+
```

## Standalone Local OCR (Without Docker Compose)

If you want to manage the local OCR container separately:

```bash
# Build the local OCR image
docker build -t mistralocr-local-ocr:latest ./local_ocr

# Run the local OCR server
docker run -d \
  --name mistralocr-local-ocr \
  -p 8000:8000 \
  -e IDLE_TIMEOUT=300 \
  -e DEFAULT_MODEL=surya \
  -v mistralocr-cache:/home/appuser/.cache \
  mistralocr-local-ocr:latest

# Test the health endpoint
curl http://localhost:8000/health
```

Then run the main app with `LOCAL_OCR_URL=http://localhost:8000` and `LOCAL_OCR_AUTO_START=false`.

## License

This project is licensed under the Mozilla Public License Version 2.0. See the [LICENSE](LICENSE) file for details.
