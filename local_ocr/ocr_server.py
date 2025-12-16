# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Local OCR Server using PaddleOCR-VL model.

This server provides a REST API for OCR processing using the PaddleOCR-VL-0.9B model.
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
from uuid import uuid4

from flask import Flask, request, jsonify, Response
from PIL import Image
import json
import fitz  # PyMuPDF for PDF handling
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "PaddlePaddle/PaddleOCR-VL")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "300"))  # 5 minutes default
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

# --- Global Model State ---
model = None
processor = None
model_lock = threading.Lock()
last_activity_time = time.time()
unload_timer = None

# Device selection
FORCE_CUDA = os.getenv("FORCE_CUDA", "false").lower() == "true"
if FORCE_CUDA and torch.cuda.is_available():
    DEVICE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Using device: {DEVICE} ({GPU_NAME}, {GPU_MEMORY:.1f}GB VRAM)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Using device: {DEVICE} ({GPU_NAME}, {GPU_MEMORY:.1f}GB VRAM)")
else:
    DEVICE = "cpu"
    print(f"Using device: {DEVICE} (GPU not available)")

# Prompts for different OCR tasks
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


def load_model():
    """Load the PaddleOCR-VL model into memory."""
    global model, processor

    with model_lock:
        if model is not None:
            print("Model already loaded")
            return True

        print(f"Loading PaddleOCR-VL model from {MODEL_PATH}...")
        start_time = time.time()

        try:
            # Use bfloat16 for GPU (memory efficient), float32 for CPU
            dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

            # Log memory before loading
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated() / (1024**3)
                print(f"  GPU memory before loading: {mem_before:.2f}GB")

            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,  # Reduce memory during loading
            ).to(DEVICE).eval()

            processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

            load_time = time.time() - start_time

            # Log memory after loading
            if DEVICE == "cuda":
                mem_after = torch.cuda.memory_allocated() / (1024**3)
                mem_peak = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"  GPU memory after loading: {mem_after:.2f}GB (peak: {mem_peak:.2f}GB)")

            print(f"Model loaded successfully in {load_time:.2f} seconds")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
            processor = None
            return False


def unload_model():
    """Unload the model from memory to free resources."""
    global model, processor

    with model_lock:
        if model is None:
            print("Model already unloaded")
            return

        # Log memory before unloading (GPU)
        if DEVICE == "cuda":
            mem_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"Unloading model to free VRAM (currently using {mem_before:.2f}GB)...")
        else:
            print("Unloading model to free memory...")

        # Move model to CPU first to free GPU memory immediately
        if DEVICE == "cuda" and model is not None:
            model.to("cpu")

        del model
        del processor
        model = None
        processor = None

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


def process_image(image: Image.Image, task: str = "ocr") -> str:
    """Process a single image with OCR."""
    global model, processor

    if model is None or processor is None:
        if not load_model():
            raise RuntimeError("Failed to load model")

    reset_idle_timer()

    prompt = PROMPTS.get(task, PROMPTS["ocr"])

    # Build messages - just text content, no image placeholder
    messages = [{"role": "user", "content": prompt}]

    with model_lock:
        print(f"    [OCR] Preparing inputs...")
        prep_start = time.time()

        # Apply chat template to get the text prompt
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process image and text together
        inputs = processor(text=[text], images=[image], return_tensors="pt")

        # Move tensors to device
        inputs = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}

        print(f"    [OCR] Inputs prepared in {time.time() - prep_start:.2f}s")
        print(f"    [OCR] Input keys: {list(inputs.keys())}")

        print(f"    [OCR] Starting inference (this may take several minutes on CPU)...")
        gen_start = time.time()

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True
            )

        print(f"    [OCR] Inference completed in {time.time() - gen_start:.2f}s")

        # Decode and extract response
        result = processor.batch_decode(generated, skip_special_tokens=True)[0]

        # Extract just the answer (after the prompt)
        if text in result:
            result = result.split(text)[-1].strip()

    return result.strip()


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
        "device": DEVICE,
        "idle_timeout": IDLE_TIMEOUT,
        "last_activity": last_activity_time,
        "time_since_activity": time.time() - last_activity_time
    })


@app.route('/load', methods=['POST'])
def load():
    """Explicitly load the model."""
    success = load_model()
    if success:
        reset_idle_timer()
        return jsonify({"status": "loaded"})
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
    - task: OCR task type (ocr, table, formula, chart) - default: ocr
    - include_images: Whether to include base64 images in response - default: true

    Returns JSON with OCR results in a format compatible with Mistral OCR response.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    task = request.form.get('task', 'ocr')
    include_images = request.form.get('include_images', 'true').lower() == 'true'

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
            markdown_content = process_image(img, task)

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
        print(f"OCR completed in {processing_time:.2f} seconds for {len(pages)} pages")

        # Return response in format compatible with Mistral OCR
        return jsonify({
            "pages": pages,
            "model": MODEL_PATH,
            "processing_time": processing_time
        })

    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp file if it has been procesed by the OCR model properly
        try:
            tmp_path.unlink()
        except:
            pass


@app.route('/ocr/stream', methods=['POST'])
def ocr_stream():
    """
    Process a PDF or image file with OCR, streaming progress updates.

    Uses newline-delimited JSON (NDJSON) to stream progress and results.

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

    task = request.form.get('task', 'ocr')
    include_images = request.form.get('include_images', 'true').lower() == 'true'

    # Save file temporarily
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = Path(tmp.name)

    def generate():
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

            # Check if model is loaded, report loading status
            if model is None:
                yield json.dumps({"type": "status", "message": "Loading OCR model"}) + "\n"

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
                markdown_content = process_image(img, task)

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
            print(f"OCR completed in {processing_time:.2f} seconds for {len(pages)} pages")

            # Send final result
            yield json.dumps({
                "type": "result",
                "pages": pages,
                "model": MODEL_PATH,
                "processing_time": processing_time
            }) + "\n"

        except Exception as e:
            print(f"Error processing file: {e}")
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
    - task: OCR task type (ocr, table, formula, chart) - default: ocr

    Returns JSON with OCR text result.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    task = data.get('task', 'ocr')
    image_data = data['image']

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
        result = process_image(image, task)

        return jsonify({
            "text": result,
            "task": task
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'false').lower() == 'true'

    print(f"Starting OCR server on {host}:{port}")
    print(f"Model: {MODEL_PATH}")
    print(f"Idle timeout: {IDLE_TIMEOUT} seconds")
    print(f"Device: {DEVICE}")

    app.run(host=host, port=port, debug=debug)
