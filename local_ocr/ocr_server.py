# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Local OCR Server using Surya OCR.

This server provides a REST API for OCR processing using Surya OCR (~300M params),
a CPU-friendly model that also works well on GPU.

It implements lazy loading and automatic unloading to minimize memory usage when idle.
"""

import os
import gc
import io
import time
import base64
import threading
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, Response
from PIL import Image
import json
import fitz  # PyMuPDF for PDF handling
import torch

app = Flask(__name__)

# --- Configuration ---
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "300"))  # 5 minutes default
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

# --- Global Model State ---
model = None
model_lock = threading.Lock()
last_activity_time = time.time()
unload_timer = None

# Device selection
print(f"[GPU Debug] PyTorch version: {torch.__version__}")
print(f"[GPU Debug] CUDA compiled: {torch.version.cuda}")
print(f"[GPU Debug] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU Debug] Device count: {torch.cuda.device_count()}")
    print(f"[GPU Debug] Device name: {torch.cuda.get_device_name(0)}")
    DEVICE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Using device: {DEVICE} ({GPU_NAME}, {GPU_MEMORY:.1f}GB VRAM)")
else:
    DEVICE = "cpu"
    print(f"Using device: {DEVICE} (GPU not available)")


def unload_model():
    """Unload the model from memory to free resources."""
    global model

    with model_lock:
        if model is None:
            print("Model already unloaded")
            return

        if DEVICE == "cuda":
            mem_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"Unloading Surya model to free VRAM (currently using {mem_before:.2f}GB)...")
        else:
            print("Unloading Surya model to free memory...")

        del model
        model = None

        # Force garbage collection
        gc.collect()

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / (1024**3)
            print(f"Model unloaded successfully (GPU memory now: {mem_after:.2f}GB)")
        else:
            print("Model unloaded successfully")


def load_model():
    """Load the Surya OCR model into memory (~300M params)."""
    global model
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    if model is not None:
        return True

    with model_lock:
        print("Loading Surya OCR model (~300M params)...")
        start_time = time.time()

        try:
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor()

            model = {
                "detection": det_predictor,
                "recognition": rec_predictor
            }

            load_time = time.time() - start_time
            print(f"Surya OCR model loaded successfully in {load_time:.2f} seconds")
            return True

        except Exception as e:
            print(f"Error loading Surya OCR model: {e}")
            import traceback
            traceback.print_exc()
            model = None
            return False


def reset_idle_timer():
    """Reset the idle timer, canceling any pending unload."""
    global last_activity_time, unload_timer

    last_activity_time = time.time()

    if unload_timer is not None:
        unload_timer.cancel()

    if IDLE_TIMEOUT > 0:
        unload_timer = threading.Timer(IDLE_TIMEOUT, check_and_unload)
        unload_timer.daemon = True
        unload_timer.start()


def check_and_unload():
    """Check if idle timeout has passed and unload if so."""
    global last_activity_time

    idle_duration = time.time() - last_activity_time
    if idle_duration >= IDLE_TIMEOUT:
        print(f"Idle timeout reached ({idle_duration:.1f}s >= {IDLE_TIMEOUT}s). Unloading model...")
        unload_model()
    else:
        remaining = IDLE_TIMEOUT - idle_duration
        timer = threading.Timer(remaining, check_and_unload)
        timer.daemon = True
        timer.start()


def process_image(image: Image.Image) -> str:
    """Process a single image with Surya OCR."""
    global model

    reset_idle_timer()

    if model is None:
        if not load_model():
            raise RuntimeError("Failed to load Surya model")

    with model_lock:
        print(f"    [Surya] Starting inference...")
        gen_start = time.time()

        det_predictor = model["detection"]
        rec_predictor = model["recognition"]

        # Run OCR using the recognition predictor with detection
        predictions = rec_predictor([image], det_predictor=det_predictor)

        # Extract text from predictions
        text_lines = []
        if predictions and len(predictions) > 0:
            ocr_result = predictions[0]
            if hasattr(ocr_result, 'text_lines'):
                for line in ocr_result.text_lines:
                    if hasattr(line, 'text'):
                        text_lines.append(line.text)
            elif hasattr(ocr_result, 'lines'):
                for line in ocr_result.lines:
                    if hasattr(line, 'text'):
                        text_lines.append(line.text)

        result_text = "\n".join(text_lines)
        print(f"    [Surya] Inference completed in {time.time() - gen_start:.2f}s")

    return result_text.strip()


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    """Convert a PDF file to a list of PIL Images."""
    images = []

    doc = fitz.open(str(pdf_path))
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)

    doc.close()
    return images


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route('/status', methods=['GET'])
def status():
    """Get model status."""
    return jsonify({
        "model_loaded": model is not None,
        "model": "surya",
        "device": DEVICE,
        "idle_timeout": IDLE_TIMEOUT,
        "last_activity": last_activity_time,
        "time_since_activity": time.time() - last_activity_time
    })


@app.route('/models', methods=['GET'])
def list_models():
    """List available models."""
    return jsonify({
        "models": [
            {
                "id": "surya",
                "name": "Surya OCR",
                "description": "Surya OCR (~300M params) - Fast, CPU and GPU friendly",
                "loaded": model is not None
            }
        ],
        "current": "surya" if model is not None else None
    })


@app.route('/load', methods=['POST'])
def load():
    """Explicitly load the model."""
    success = load_model()
    if success:
        reset_idle_timer()
        return jsonify({"status": "loaded", "model": "surya"})
    else:
        return jsonify({"status": "error", "message": "Failed to load model"}), 500


@app.route('/unload', methods=['POST'])
def unload():
    """Explicitly unload the model."""
    unload_model()
    return jsonify({"status": "unloaded"})


@app.route('/ocr', methods=['POST'])
def ocr():
    """
    Process a PDF or image file with OCR.

    Accepts multipart/form-data with:
    - file: The PDF or image file to process
    - include_images: Whether to include base64 images in response - default: true

    Returns JSON with OCR results in a format compatible with Mistral OCR response.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    include_images = request.form.get('include_images', 'true').lower() == 'true'

    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        start_time = time.time()

        if suffix == '.pdf':
            print(f"Processing PDF: {file.filename}")
            images = pdf_to_images(tmp_path)
        elif suffix in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']:
            print(f"Processing image: {file.filename}")
            images = [Image.open(tmp_path).convert('RGB')]
        else:
            return jsonify({"error": f"Unsupported file type: {suffix}"}), 400

        pages = []
        for page_idx, img in enumerate(images):
            print(f"  Processing page {page_idx + 1}/{len(images)}...")

            markdown_content = process_image(img)

            page_data = {
                "index": page_idx,
                "markdown": markdown_content,
                "images": []
            }

            if include_images:
                img_id = f"page_{page_idx + 1}_img"
                img_base64 = image_to_base64(img.convert('RGB'), "PNG")
                page_data["images"].append({
                    "id": img_id,
                    "image_base64": f"data:image/png;base64,{img_base64}"
                })
                if f"![" not in page_data["markdown"]:
                    page_data["markdown"] = f"![{img_id}]({img_id})\n\n{page_data['markdown']}"

            pages.append(page_data)

        processing_time = time.time() - start_time
        print(f"OCR completed in {processing_time:.2f} seconds for {len(pages)} pages")

        return jsonify({
            "pages": pages,
            "model": "surya",
            "processing_time": processing_time
        })

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            tmp_path.unlink()
        except:
            pass


@app.route('/ocr/stream', methods=['POST'])
def ocr_stream():
    """
    Process a PDF or image file with OCR, streaming progress updates.

    Uses newline-delimited JSON (NDJSON) to stream progress and results.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    include_images = request.form.get('include_images', 'true').lower() == 'true'

    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = Path(tmp.name)

    def generate():
        try:
            start_time = time.time()

            yield json.dumps({"type": "status", "message": "Loading file"}) + "\n"

            if suffix == '.pdf':
                print(f"Processing PDF: {file.filename}")
                yield json.dumps({"type": "status", "message": "Converting PDF to images"}) + "\n"
                images = pdf_to_images(tmp_path)
            elif suffix in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']:
                print(f"Processing image: {file.filename}")
                images = [Image.open(tmp_path).convert('RGB')]
            else:
                yield json.dumps({"type": "error", "message": f"Unsupported file type: {suffix}"}) + "\n"
                return

            total_pages = len(images)
            yield json.dumps({"type": "status", "message": f"Found {total_pages} pages"}) + "\n"

            if model is None:
                yield json.dumps({"type": "status", "message": "Loading Surya OCR model"}) + "\n"

            pages = []
            for page_idx, img in enumerate(images):
                page_num = page_idx + 1
                print(f"  Processing page {page_num}/{total_pages}...")

                yield json.dumps({
                    "type": "progress",
                    "page": page_num,
                    "total": total_pages,
                    "message": f"Processing page {page_num}/{total_pages}"
                }) + "\n"

                markdown_content = process_image(img)

                page_data = {
                    "index": page_idx,
                    "markdown": markdown_content,
                    "images": []
                }

                if include_images:
                    img_id = f"page_{page_num}_img"
                    img_base64 = image_to_base64(img.convert('RGB'), "PNG")
                    page_data["images"].append({
                        "id": img_id,
                        "image_base64": f"data:image/png;base64,{img_base64}"
                    })
                    if f"![" not in page_data["markdown"]:
                        page_data["markdown"] = f"![{img_id}]({img_id})\n\n{page_data['markdown']}"

                pages.append(page_data)

            processing_time = time.time() - start_time
            print(f"OCR completed in {processing_time:.2f} seconds for {len(pages)} pages")

            yield json.dumps({
                "type": "result",
                "pages": pages,
                "model": "surya",
                "processing_time": processing_time
            }) + "\n"

        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

        finally:
            try:
                tmp_path.unlink()
            except:
                pass

    return Response(generate(), mimetype='application/x-ndjson')


@app.route('/ocr/image', methods=['POST'])
def ocr_image():
    """
    Process a single image with OCR.
    Accepts base64-encoded image data in JSON.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_data = data['image']

    if image_data.startswith('data:'):
        try:
            image_data = image_data.split(',', 1)[1]
        except IndexError:
            return jsonify({"error": "Malformed data URI"}), 400

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        result = process_image(image)

        return jsonify({
            "text": result,
            "model": "surya"
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'false').lower() == 'true'

    print(f"Starting OCR server on {host}:{port}")
    print(f"Model: Surya OCR (~300M params)")
    print(f"Idle timeout: {IDLE_TIMEOUT} seconds")
    print(f"Device: {DEVICE}")

    app.run(host=host, port=port, debug=debug)
