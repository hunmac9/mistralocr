# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Local OCR Server with Multiple Backends.

This server provides a REST API for OCR processing with two backend options:
- Surya OCR (~300M params) - Fast, CPU and GPU friendly
- PaddleOCR-VL (0.9B params) - SOTA accuracy, handles tables/formulas/charts

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
DEFAULT_BACKEND = os.getenv("OCR_BACKEND", "surya")  # "surya" or "paddleocr-vl"
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

# --- Global Model State ---
surya_model = None
paddleocr_model = None
paddleocr_processor = None
current_backend = DEFAULT_BACKEND
model_lock = threading.Lock()
last_activity_time = time.time()
unload_timer = None
active_requests = 0  # Count of requests currently processing
active_requests_lock = threading.Lock()

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


# ============================================================================
# Surya OCR Backend
# ============================================================================

def load_surya():
    """Load the Surya OCR model into memory (~300M params)."""
    global surya_model
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    if surya_model is not None:
        return True

    with model_lock:
        if surya_model is not None:
            return True

        print("Loading Surya OCR model (~300M params)...")
        start_time = time.time()

        try:
            # surya-ocr 0.17+ API: FoundationPredictor shared across predictors
            foundation_predictor = FoundationPredictor()
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor(foundation_predictor)

            surya_model = {
                "detection": det_predictor,
                "recognition": rec_predictor,
                "foundation": foundation_predictor  # Keep reference to prevent GC
            }

            load_time = time.time() - start_time
            print(f"Surya OCR model loaded successfully in {load_time:.2f} seconds")
            # Reset idle timer after long model loading
            reset_idle_timer()
            return True

        except Exception as e:
            print(f"Error loading Surya OCR model: {e}")
            import traceback
            traceback.print_exc()
            surya_model = None
            return False


def unload_surya():
    """Unload the Surya model from memory."""
    global surya_model

    with model_lock:
        if surya_model is None:
            return

        if DEVICE == "cuda":
            mem_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"Unloading Surya model (using {mem_before:.2f}GB)...")
        else:
            print("Unloading Surya model...")

        del surya_model
        surya_model = None


def process_image_surya(image: Image.Image) -> str:
    """Process a single image with Surya OCR."""
    global surya_model

    if surya_model is None:
        if not load_surya():
            raise RuntimeError("Failed to load Surya model")

    with model_lock:
        print(f"    [Surya] Starting inference...")
        gen_start = time.time()

        det_predictor = surya_model["detection"]
        rec_predictor = surya_model["recognition"]

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


# ============================================================================
# PaddleOCR-VL Backend
# ============================================================================

def load_paddleocr_vl():
    """Load the PaddleOCR-VL model (0.9B params) via transformers 5.0+."""
    global paddleocr_model, paddleocr_processor
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if paddleocr_model is not None:
        return True

    with model_lock:
        if paddleocr_model is not None:
            return True

        print("Loading PaddleOCR-VL model (0.9B params)...")
        start_time = time.time()

        try:
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
            }

            # Use flash-attention if available for faster inference
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("  Using flash-attention for faster inference")
            except ImportError:
                print("  Flash-attention not available, using standard attention")

            # Use AutoModelForImageTextToText for transformers 5.0+ native support
            paddleocr_model = AutoModelForImageTextToText.from_pretrained(
                "PaddlePaddle/PaddleOCR-VL",
                **model_kwargs
            ).to(DEVICE).eval()

            paddleocr_processor = AutoProcessor.from_pretrained(
                "PaddlePaddle/PaddleOCR-VL"
            )

            load_time = time.time() - start_time
            if DEVICE == "cuda":
                mem_used = torch.cuda.memory_allocated() / (1024**3)
                print(f"PaddleOCR-VL loaded in {load_time:.2f}s (using {mem_used:.2f}GB VRAM)")
            else:
                print(f"PaddleOCR-VL loaded in {load_time:.2f}s")
            # Reset idle timer after long model loading
            reset_idle_timer()
            return True

        except Exception as e:
            print(f"Error loading PaddleOCR-VL model: {e}")
            import traceback
            traceback.print_exc()
            paddleocr_model = None
            paddleocr_processor = None
            return False


def unload_paddleocr_vl():
    """Unload the PaddleOCR-VL model from memory."""
    global paddleocr_model, paddleocr_processor

    with model_lock:
        if paddleocr_model is None:
            return

        if DEVICE == "cuda":
            mem_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"Unloading PaddleOCR-VL (using {mem_before:.2f}GB)...")
        else:
            print("Unloading PaddleOCR-VL...")

        del paddleocr_model
        del paddleocr_processor
        paddleocr_model = None
        paddleocr_processor = None


def process_image_paddleocr(image: Image.Image) -> str:
    """Process a single image with PaddleOCR-VL."""
    global paddleocr_model, paddleocr_processor

    if paddleocr_model is None:
        if not load_paddleocr_vl():
            raise RuntimeError("Failed to load PaddleOCR-VL model")

    with model_lock:
        print(f"    [PaddleOCR-VL] Starting inference...")
        gen_start = time.time()

        # Prepare the message with image and OCR prompt (transformers 5.0+ format)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "OCR:"},
                ]
            }
        ]

        # Process the input using apply_chat_template
        inputs = paddleocr_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(DEVICE)

        # Generate the OCR output
        with torch.inference_mode():
            outputs = paddleocr_model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                use_cache=True,
            )

        # Decode result - trim the input tokens from output
        generated_ids_trimmed = outputs[0][inputs["input_ids"].shape[-1]:]
        result = paddleocr_processor.decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(f"    [PaddleOCR-VL] Inference completed in {time.time() - gen_start:.2f}s")

    return result.strip()


# ============================================================================
# Common Functions
# ============================================================================

def unload_all_models():
    """Unload all models from memory to free resources."""
    print("Unloading all models...")
    unload_surya()
    unload_paddleocr_vl()

    # Force garbage collection
    gc.collect()

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / (1024**3)
        print(f"All models unloaded (GPU memory now: {mem_after:.2f}GB)")
    else:
        print("All models unloaded")


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
    global last_activity_time, unload_timer

    # Don't unload if there are active requests processing
    with active_requests_lock:
        if active_requests > 0:
            print(f"Skipping unload check - {active_requests} active request(s)")
            # Reschedule check for later
            unload_timer = threading.Timer(IDLE_TIMEOUT, check_and_unload)
            unload_timer.daemon = True
            unload_timer.start()
            return

    idle_duration = time.time() - last_activity_time
    if idle_duration >= IDLE_TIMEOUT:
        print(f"Idle timeout reached ({idle_duration:.1f}s >= {IDLE_TIMEOUT}s). Unloading models...")
        unload_all_models()
        unload_timer = None
    else:
        remaining = IDLE_TIMEOUT - idle_duration
        # Store timer in global so it can be cancelled by reset_idle_timer()
        unload_timer = threading.Timer(remaining, check_and_unload)
        unload_timer.daemon = True
        unload_timer.start()


def process_image(image: Image.Image, backend: str = None) -> str:
    """Process a single image with the specified or default OCR backend."""
    global current_backend

    reset_idle_timer()
    backend = backend or current_backend

    try:
        if backend == "paddleocr-vl":
            return process_image_paddleocr(image)
        else:  # surya (default)
            return process_image_surya(image)
    finally:
        # Reset timer after processing completes to ensure timeout
        # doesn't fire during next page's inference
        reset_idle_timer()


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


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route('/status', methods=['GET'])
def status():
    """Get model status."""
    return jsonify({
        "model_loaded": surya_model is not None or paddleocr_model is not None,
        "current_backend": current_backend,
        "device": DEVICE,
        "idle_timeout": IDLE_TIMEOUT,
        "last_activity": last_activity_time,
        "time_since_activity": time.time() - last_activity_time,
        "backends": {
            "surya": {"loaded": surya_model is not None},
            "paddleocr-vl": {"loaded": paddleocr_model is not None},
        }
    })


@app.route('/models', methods=['GET'])
def list_models():
    """List available models/backends."""
    return jsonify({
        "models": [
            {
                "id": "surya",
                "name": "Surya OCR",
                "description": "Surya OCR (~300M params) - Fast, CPU and GPU friendly",
                "loaded": surya_model is not None
            },
            {
                "id": "paddleocr-vl",
                "name": "PaddleOCR-VL",
                "description": "PaddleOCR-VL (0.9B params) - SOTA accuracy, tables/formulas/charts",
                "loaded": paddleocr_model is not None
            }
        ],
        "current": current_backend
    })


@app.route('/backend', methods=['GET'])
def get_backend():
    """Get current OCR backend."""
    return jsonify({
        "current": current_backend,
        "available": ["surya", "paddleocr-vl"],
        "loaded": {
            "surya": surya_model is not None,
            "paddleocr-vl": paddleocr_model is not None,
        }
    })


@app.route('/backend', methods=['POST'])
def set_backend():
    """Switch OCR backend."""
    global current_backend
    data = request.get_json() or {}
    backend = data.get('backend', 'surya')

    if backend not in ["surya", "paddleocr-vl"]:
        return jsonify({"error": f"Unknown backend: {backend}"}), 400

    current_backend = backend
    print(f"Switched OCR backend to: {current_backend}")
    return jsonify({"backend": current_backend})


@app.route('/load', methods=['POST'])
def load():
    """Explicitly load the current backend's model."""
    backend = request.get_json().get('backend', current_backend) if request.is_json else current_backend

    if backend == "paddleocr-vl":
        success = load_paddleocr_vl()
        model_name = "PaddleOCR-VL"
    else:
        success = load_surya()
        model_name = "Surya"

    if success:
        reset_idle_timer()
        return jsonify({"status": "loaded", "model": model_name, "backend": backend})
    else:
        return jsonify({"status": "error", "message": f"Failed to load {model_name}"}), 500


@app.route('/unload', methods=['POST'])
def unload():
    """Explicitly unload all models."""
    unload_all_models()
    return jsonify({"status": "unloaded"})


@app.route('/ocr', methods=['POST'])
def ocr():
    """
    Process a PDF or image file with OCR.

    Accepts multipart/form-data with:
    - file: The PDF or image file to process
    - include_images: Whether to include base64 images in response - default: true
    - backend: OCR backend to use - "surya" or "paddleocr-vl" (optional, uses current default)

    Returns JSON with OCR results in a format compatible with Mistral OCR response.
    """
    global active_requests

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    include_images = request.form.get('include_images', 'true').lower() == 'true'
    backend = request.form.get('backend', current_backend)

    # Validate backend
    if backend not in ["surya", "paddleocr-vl"]:
        return jsonify({"error": f"Unknown backend: {backend}"}), 400

    suffix = Path(file.filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = Path(tmp.name)

    # Track active request to prevent model unloading during processing
    with active_requests_lock:
        active_requests += 1
        print(f"Active requests: {active_requests}")

    try:
        start_time = time.time()

        if suffix == '.pdf':
            print(f"Processing PDF: {file.filename} (backend: {backend})")
            images = pdf_to_images(tmp_path)
        elif suffix in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']:
            print(f"Processing image: {file.filename} (backend: {backend})")
            images = [Image.open(tmp_path).convert('RGB')]
        else:
            return jsonify({"error": f"Unsupported file type: {suffix}"}), 400

        pages = []
        for page_idx, img in enumerate(images):
            print(f"  Processing page {page_idx + 1}/{len(images)}...")

            markdown_content = process_image(img, backend=backend)

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
            "model": backend,
            "processing_time": processing_time
        })

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Decrement active requests counter
        with active_requests_lock:
            active_requests -= 1
            print(f"Active requests: {active_requests}")
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
    global active_requests

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    include_images = request.form.get('include_images', 'true').lower() == 'true'
    backend = request.form.get('backend', current_backend)

    # Validate backend
    if backend not in ["surya", "paddleocr-vl"]:
        return jsonify({"error": f"Unknown backend: {backend}"}), 400

    suffix = Path(file.filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = Path(tmp.name)

    def generate():
        global active_requests

        # Track active request to prevent model unloading during processing
        with active_requests_lock:
            active_requests += 1
            print(f"Active requests: {active_requests}")

        try:
            start_time = time.time()

            yield json.dumps({"type": "status", "message": "Loading file"}) + "\n"

            if suffix == '.pdf':
                print(f"Processing PDF: {file.filename} (backend: {backend})")
                yield json.dumps({"type": "status", "message": "Converting PDF to images"}) + "\n"
                images = pdf_to_images(tmp_path)
            elif suffix in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']:
                print(f"Processing image: {file.filename} (backend: {backend})")
                images = [Image.open(tmp_path).convert('RGB')]
            else:
                yield json.dumps({"type": "error", "message": f"Unsupported file type: {suffix}"}) + "\n"
                return

            total_pages = len(images)
            yield json.dumps({"type": "status", "message": f"Found {total_pages} pages"}) + "\n"

            # Check if model needs to be loaded
            if backend == "paddleocr-vl" and paddleocr_model is None:
                yield json.dumps({"type": "status", "message": "Loading PaddleOCR-VL model"}) + "\n"
            elif backend == "surya" and surya_model is None:
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

                markdown_content = process_image(img, backend=backend)

                # Send page completion message to reset client timeouts
                yield json.dumps({
                    "type": "page_complete",
                    "page": page_num,
                    "total": total_pages,
                    "message": f"Completed page {page_num}/{total_pages}"
                }) + "\n"

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
                "model": backend,
                "processing_time": processing_time
            }) + "\n"

        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

        finally:
            # Decrement active requests counter
            with active_requests_lock:
                active_requests -= 1
                print(f"Active requests: {active_requests}")
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
    backend = data.get('backend', current_backend)

    # Validate backend
    if backend not in ["surya", "paddleocr-vl"]:
        return jsonify({"error": f"Unknown backend: {backend}"}), 400

    if image_data.startswith('data:'):
        try:
            image_data = image_data.split(',', 1)[1]
        except IndexError:
            return jsonify({"error": "Malformed data URI"}), 400

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        result = process_image(image, backend=backend)

        return jsonify({
            "text": result,
            "model": backend
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
    print(f"Available backends: Surya OCR, PaddleOCR-VL")
    print(f"Default backend: {DEFAULT_BACKEND}")
    print(f"Idle timeout: {IDLE_TIMEOUT} seconds")
    print(f"Device: {DEVICE}")

    app.run(host=host, port=port, debug=debug)
