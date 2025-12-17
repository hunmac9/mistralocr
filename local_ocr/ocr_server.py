# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Local OCR Server supporting multiple OCR models.

This server provides a REST API for OCR processing using:
- Surya OCR (~300M params) - CPU-friendly, recommended for CPU-only deployment
- Chandra OCR model (9B params) - GPU required

It implements lazy loading and automatic unloading to minimize memory usage when idle.
Models are mutually exclusive - only one can be loaded at a time.
"""

import os
import gc
import io
import time
import base64
import threading
import tempfile
from pathlib import Path
from enum import Enum

from flask import Flask, request, jsonify, Response
from PIL import Image
import json
import fitz  # PyMuPDF for PDF handling
import torch

app = Flask(__name__)

# --- Configuration ---
# Model selection: "surya" (CPU-friendly, default), "chandra" (GPU required)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "surya")  # Default to Surya for CPU compatibility
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "300"))  # 5 minutes default
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

# Model paths
CHANDRA_MODEL_PATH = os.getenv("CHANDRA_MODEL_PATH", "datalab-to/chandra")
# Surya uses built-in model paths from the surya-ocr package


class ModelType(Enum):
    CHANDRA = "chandra"
    SURYA = "surya"


# --- Global Model State ---
current_model_type: ModelType | None = None
model = None
processor = None
model_lock = threading.Lock()
last_activity_time = time.time()
unload_timer = None

# Device selection - Debug GPU availability
print(f"[GPU Debug] PyTorch version: {torch.__version__}")
print(f"[GPU Debug] CUDA compiled: {torch.version.cuda}")
print(f"[GPU Debug] CUDA available: {torch.cuda.is_available()}")
print(f"[GPU Debug] cuDNN available: {torch.backends.cudnn.is_available() if hasattr(torch.backends, 'cudnn') else 'N/A'}")
if not torch.cuda.is_available():
    # Try to get more info about why CUDA isn't available
    try:
        torch.cuda.init()
    except Exception as e:
        print(f"[GPU Debug] CUDA init error: {e}")
else:
    print(f"[GPU Debug] Device count: {torch.cuda.device_count()}")
    print(f"[GPU Debug] Device name: {torch.cuda.get_device_name(0)}")

FORCE_CUDA = os.getenv("FORCE_CUDA", "false").lower() == "true"
if torch.cuda.is_available():
    DEVICE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Using device: {DEVICE} ({GPU_NAME}, {GPU_MEMORY:.1f}GB VRAM)")
else:
    DEVICE = "cpu"
    print(f"Using device: {DEVICE} (GPU not available)")
    if FORCE_CUDA:
        print("[WARNING] FORCE_CUDA=true but CUDA is not available!")


def unload_model():
    """Unload the current model from memory to free resources."""
    global model, processor, current_model_type

    with model_lock:
        if model is None:
            print("Model already unloaded")
            return

        model_name = current_model_type.value if current_model_type else "unknown"

        # Log memory before unloading (GPU)
        if DEVICE == "cuda":
            mem_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"Unloading {model_name} model to free VRAM (currently using {mem_before:.2f}GB)...")
        else:
            print(f"Unloading {model_name} model to free memory...")

        # Move model to CPU first to free GPU memory immediately
        if DEVICE == "cuda" and model is not None:
            try:
                model.to("cpu")
            except:
                pass

        del model
        del processor
        model = None
        processor = None
        current_model_type = None

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



def load_chandra_model():
    """Load the Chandra OCR model into memory."""
    global model, processor, current_model_type
    from transformers import AutoModel, AutoProcessor

    # If another model is loaded, unload it first
    if current_model_type is not None:
        unload_model()

    with model_lock:
        print(f"Loading Chandra OCR model from {CHANDRA_MODEL_PATH}...")
        start_time = time.time()

        try:
            # Log memory before loading
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated() / (1024**3)
                print(f"  GPU memory before loading: {mem_before:.2f}GB")

            # Chandra uses AutoModel, not AutoModelForCausalLM
            if DEVICE == "cuda":
                print(f"  Loading Chandra model to GPU...")
                model = AutoModel.from_pretrained(
                    CHANDRA_MODEL_PATH,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                ).eval()
            else:
                print(f"  Loading Chandra model to CPU...")
                model = AutoModel.from_pretrained(
                    CHANDRA_MODEL_PATH,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                ).eval()

            processor = AutoProcessor.from_pretrained(CHANDRA_MODEL_PATH, trust_remote_code=True)
            # Attach processor to model as Chandra expects
            model.processor = processor
            current_model_type = ModelType.CHANDRA

            load_time = time.time() - start_time

            # Log memory after loading
            if DEVICE == "cuda":
                mem_after = torch.cuda.memory_allocated() / (1024**3)
                mem_peak = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"  GPU memory after loading: {mem_after:.2f}GB (peak: {mem_peak:.2f}GB)")

            print(f"Chandra OCR model loaded successfully in {load_time:.2f} seconds")
            return True

        except Exception as e:
            print(f"Error loading Chandra OCR model: {e}")
            import traceback
            traceback.print_exc()
            model = None
            processor = None
            current_model_type = None
            return False


def load_surya_model():
    """Load the Surya OCR model into memory (~300M params, CPU-friendly)."""
    global model, processor, current_model_type
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor

    # If another model is loaded, unload it first
    if current_model_type is not None:
        unload_model()

    with model_lock:
        print("Loading Surya OCR model (CPU-friendly, ~300M params)...")
        start_time = time.time()

        try:
            # Surya requires FoundationPredictor for RecognitionPredictor
            foundation_predictor = FoundationPredictor()
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor(foundation_predictor)

            # Store predictors for easy access
            model = {
                "foundation": foundation_predictor,
                "detection": det_predictor,
                "recognition": rec_predictor
            }
            processor = None  # Surya doesn't use a separate processor
            current_model_type = ModelType.SURYA

            load_time = time.time() - start_time
            print(f"Surya OCR model loaded successfully in {load_time:.2f} seconds")
            return True

        except Exception as e:
            print(f"Error loading Surya OCR model: {e}")
            import traceback
            traceback.print_exc()
            model = None
            processor = None
            current_model_type = None
            return False


def load_model(model_type: ModelType = None):
    """Load the specified model (or default if not specified)."""
    if model_type is None:
        model_type = ModelType(DEFAULT_MODEL)

    if model_type == ModelType.CHANDRA:
        return load_chandra_model()
    elif model_type == ModelType.SURYA:
        return load_surya_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
        # Reschedule check
        remaining = IDLE_TIMEOUT - idle_duration
        timer = threading.Timer(remaining, check_and_unload)
        timer.daemon = True
        timer.start()



def process_image_chandra(image: Image.Image) -> str:
    """Process a single image with Chandra OCR."""
    global model
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown

    reset_idle_timer()

    with model_lock:
        print(f"    [Chandra] Starting inference...")
        gen_start = time.time()

        batch = [
            BatchInputItem(
                image=image,
                prompt_type="ocr_layout"
            )
        ]

        result = generate_hf(batch, model)[0]
        markdown = parse_markdown(result.raw)

        print(f"    [Chandra] Inference completed in {time.time() - gen_start:.2f}s")

    return markdown.strip()


def process_image_surya(image: Image.Image) -> str:
    """Process a single image with Surya OCR (~300M params, CPU-friendly)."""
    global model

    reset_idle_timer()

    with model_lock:
        print(f"    [Surya] Starting inference...")
        gen_start = time.time()

        det_predictor = model["detection"]
        rec_predictor = model["recognition"]

        # Run OCR using the recognition predictor with detection
        # The recognition predictor uses the detection predictor internally
        predictions = rec_predictor([image], det_predictor=det_predictor)

        # Extract text from predictions
        # Surya returns a list of OCRResult objects, one per image
        text_lines = []
        if predictions and len(predictions) > 0:
            ocr_result = predictions[0]
            # OCRResult has text_lines attribute containing recognized text
            if hasattr(ocr_result, 'text_lines'):
                for line in ocr_result.text_lines:
                    if hasattr(line, 'text'):
                        text_lines.append(line.text)
            # Fallback: some versions may have different attribute names
            elif hasattr(ocr_result, 'lines'):
                for line in ocr_result.lines:
                    if hasattr(line, 'text'):
                        text_lines.append(line.text)

        result_text = "\n".join(text_lines)
        print(f"    [Surya] Inference completed in {time.time() - gen_start:.2f}s")

    return result_text.strip()


def process_image(image: Image.Image, model_type: ModelType = None) -> str:
    """Process a single image with OCR using the specified or current model."""
    global model, processor, current_model_type

    # Determine which model to use
    if model_type is None:
        if current_model_type is not None:
            model_type = current_model_type
        else:
            model_type = ModelType(DEFAULT_MODEL)

    # Load model if not loaded or if different model requested
    if model is None or current_model_type != model_type:
        if not load_model(model_type):
            raise RuntimeError(f"Failed to load {model_type.value} model")

    # Process with the appropriate model
    if current_model_type == ModelType.CHANDRA:
        return process_image_chandra(image)
    elif current_model_type == ModelType.SURYA:
        return process_image_surya(image)
    else:
        raise RuntimeError(f"Unknown model type: {current_model_type}")


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    """Convert a PDF file to a list of PIL Images."""
    images = []

    doc = fitz.open(str(pdf_path))
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Use a matrix to control the resolution
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
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
        "current_model": current_model_type.value if current_model_type else None,
        "available_models": ["surya", "chandra"],
        "default_model": DEFAULT_MODEL,
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
                "description": "Surya OCR (~300M params) - CPU-friendly, recommended for CPU-only deployment",
                "loaded": current_model_type == ModelType.SURYA
            },

            {
                "id": "chandra",
                "name": "Chandra OCR",
                "description": "Chandra OCR (9B params) - layout preservation (GPU required)",
                "loaded": current_model_type == ModelType.CHANDRA
            }
        ],
        "current": current_model_type.value if current_model_type else None
    })


@app.route('/load', methods=['POST'])
def load():
    """Explicitly load a model."""
    data = request.get_json() or {}
    model_name = data.get('model', DEFAULT_MODEL)

    try:
        model_type = ModelType(model_name)
    except ValueError:
        return jsonify({"status": "error", "message": f"Unknown model: {model_name}"}), 400

    success = load_model(model_type)
    if success:
        reset_idle_timer()
        return jsonify({"status": "loaded", "model": model_name})
    else:
        return jsonify({"status": "error", "message": f"Failed to load model: {model_name}"}), 500


@app.route('/unload', methods=['POST'])
def unload():
    """Explicitly unload the current model."""
    unload_model()
    return jsonify({"status": "unloaded"})


@app.route('/switch', methods=['POST'])
def switch_model():
    """Switch to a different model (unloads current, loads new)."""
    data = request.get_json() or {}
    model_name = data.get('model')

    if not model_name:
        return jsonify({"status": "error", "message": "Model name required"}), 400

    try:
        model_type = ModelType(model_name)
    except ValueError:
        return jsonify({"status": "error", "message": f"Unknown model: {model_name}"}), 400

    # If same model, just return success
    if current_model_type == model_type:
        return jsonify({"status": "ok", "message": f"Model {model_name} already loaded"})

    success = load_model(model_type)
    if success:
        reset_idle_timer()
        return jsonify({"status": "switched", "model": model_name})
    else:
        return jsonify({"status": "error", "message": f"Failed to switch to model: {model_name}"}), 500


@app.route('/ocr', methods=['POST'])
def ocr():
    """
    Process a PDF or image file with OCR.

    Accepts multipart/form-data with:
    - file: The PDF or image file to process
    - model: Model to use (surya, chandra) - default: current or default model
    - include_images: Whether to include base64 images in response - default: true

    Returns JSON with OCR results in a format compatible with Mistral OCR response.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    include_images = request.form.get('include_images', 'true').lower() == 'true'
    model_name = request.form.get('model')

    # Determine model type
    model_type = None
    if model_name:
        try:
            model_type = ModelType(model_name)
        except ValueError:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

    # Save file temporarily
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        start_time = time.time()

        # Convert PDF to images or load image directly
        if suffix == '.pdf':
            print(f"Processing PDF: {file.filename}")
            images = pdf_to_images(tmp_path)
        elif suffix in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']:
            print(f"Processing image: {file.filename}")
            images = [Image.open(tmp_path).convert('RGB')]
        else:
            return jsonify({"error": f"Unsupported file type: {suffix}"}), 400

        # Process each page/image
        pages = []
        for page_idx, img in enumerate(images):
            print(f"  Processing page {page_idx + 1}/{len(images)}...")

            # Run OCR on the page
            markdown_content = process_image(img, model_type)

            # Prepare page response
            page_data = {
                "index": page_idx,
                "markdown": markdown_content,
                "images": []
            }

            # Include the page image if requested
            if include_images:
                img_id = f"page_{page_idx + 1}_img"
                img_base64 = image_to_base64(img.convert('RGB'), "PNG")
                page_data["images"].append({
                    "id": img_id,
                    "image_base64": f"data:image/png;base64,{img_base64}"
                })
                # Add image reference to markdown if not already present
                if f"![" not in page_data["markdown"]:
                    page_data["markdown"] = f"![{img_id}]({img_id})\n\n{page_data['markdown']}"

            pages.append(page_data)

        processing_time = time.time() - start_time
        used_model = current_model_type.value if current_model_type else DEFAULT_MODEL
        print(f"OCR completed in {processing_time:.2f} seconds for {len(pages)} pages using {used_model}")

        # Return response in format compatible with Mistral OCR
        return jsonify({
            "pages": pages,
            "model": used_model,
            "processing_time": processing_time
        })

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp file
        try:
            tmp_path.unlink()
        except:
            pass


@app.route('/ocr/stream', methods=['POST'])
def ocr_stream():
    """
    Process a PDF or image file with OCR, streaming progress updates.

    Uses newline-delimited JSON (NDJSON) to stream progress and results.

    Form parameters:
    - file: The PDF or image file to process
    - model: Model to use (surya, chandra) - default: current or default model
    - include_images: Whether to include base64 images in response - default: true

    Progress messages:
    - {"type": "status", "message": "..."}
    - {"type": "progress", "page": 1, "total": 10, "message": "Processing page 1/10"}
    - {"type": "result", "pages": [...], "model": "...", "processing_time": ...}
    - {"type": "error", "message": "..."}
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    include_images = request.form.get('include_images', 'true').lower() == 'true'
    model_name = request.form.get('model')

    # Determine model type
    model_type = None
    if model_name:
        try:
            model_type = ModelType(model_name)
        except ValueError:
            return Response(
                json.dumps({"type": "error", "message": f"Unknown model: {model_name}"}) + "\n",
                mimetype='application/x-ndjson'
            )

    # Save file temporarily
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = Path(tmp.name)

    def generate():
        global current_model_type
        try:
            start_time = time.time()

            # Send initial status
            yield json.dumps({"type": "status", "message": "Loading file"}) + "\n"

            # Convert PDF to images or load image directly
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

            # Determine target model
            target_model = model_type
            if target_model is None:
                if current_model_type is not None:
                    target_model = current_model_type
                else:
                    target_model = ModelType(DEFAULT_MODEL)

            # Check if we need to load/switch models
            if model is None or current_model_type != target_model:
                model_display = "Surya OCR" if target_model == ModelType.SURYA else "Chandra OCR"
                yield json.dumps({"type": "status", "message": f"Loading {model_display} model"}) + "\n"

            # Process each page/image
            pages = []
            for page_idx, img in enumerate(images):
                page_num = page_idx + 1
                print(f"  Processing page {page_num}/{total_pages}...")

                # Send progress update
                yield json.dumps({
                    "type": "progress",
                    "page": page_num,
                    "total": total_pages,
                    "message": f"Processing page {page_num}/{total_pages}"
                }) + "\n"

                # Run OCR on the page
                markdown_content = process_image(img, target_model)

                # Prepare page response
                page_data = {
                    "index": page_idx,
                    "markdown": markdown_content,
                    "images": []
                }

                # Include the page image if requested
                if include_images:
                    img_id = f"page_{page_num}_img"
                    img_base64 = image_to_base64(img.convert('RGB'), "PNG")
                    page_data["images"].append({
                        "id": img_id,
                        "image_base64": f"data:image/png;base64,{img_base64}"
                    })
                    # Add image reference to markdown if not already present
                    if f"![" not in page_data["markdown"]:
                        page_data["markdown"] = f"![{img_id}]({img_id})\n\n{page_data['markdown']}"

                pages.append(page_data)

            processing_time = time.time() - start_time
            used_model = current_model_type.value if current_model_type else DEFAULT_MODEL
            print(f"OCR completed in {processing_time:.2f} seconds for {len(pages)} pages using {used_model}")

            # Send final result
            yield json.dumps({
                "type": "result",
                "pages": pages,
                "model": used_model,
                "processing_time": processing_time
            }) + "\n"

        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

        finally:
            # Clean up temp file
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

    JSON body:
    - image: Base64-encoded image data (with or without data URI prefix)
    - model: Model to use (surya, chandra) - default: current or default model

    Returns JSON with OCR text result.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_data = data['image']
    model_name = data.get('model')

    # Determine model type
    model_type = None
    if model_name:
        try:
            model_type = ModelType(model_name)
        except ValueError:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

    # Handle data URI prefix
    if image_data.startswith('data:'):
        try:
            image_data = image_data.split(',', 1)[1]
        except IndexError:
            return jsonify({"error": "Malformed data URI"}), 400

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Process with OCR
        result = process_image(image, model_type)

        return jsonify({
            "text": result,
            "model": current_model_type.value if current_model_type else DEFAULT_MODEL
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
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Available models: Surya OCR (CPU-friendly), Chandra OCR ({CHANDRA_MODEL_PATH})")
    print(f"Idle timeout: {IDLE_TIMEOUT} seconds")
    print(f"Device: {DEVICE}")

    app.run(host=host, port=port, debug=debug)
