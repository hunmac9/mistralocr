# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import json
import base64
import shutil
import zipfile
import queue # Added for SSE
from pathlib import Path
from uuid import uuid4
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for, redirect, Response # Added Response for SSE
import threading
import time
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from PyPDF2 import PdfReader, PdfWriter
from PIL import Image
import io

load_dotenv() # Load environment variables from .env file

app = Flask(__name__)

# --- Configuration ---
FLASK_PORT_INT = int(os.getenv('FLASK_PORT', 5009)) # Get port for default SERVER_NAME

# Parse SERVER_NAME env var to handle full URLs
raw_server_name = os.getenv('SERVER_NAME', f'localhost:{FLASK_PORT_INT}')
if raw_server_name.startswith('http://'):
    app.config['PREFERRED_URL_SCHEME'] = 'http'
    raw_server_name = raw_server_name[len('http://'):]
elif raw_server_name.startswith('https://'):
    app.config['PREFERRED_URL_SCHEME'] = 'https'
    raw_server_name = raw_server_name[len('https://'):]
else:
    # Default scheme if not specified
    app.config['PREFERRED_URL_SCHEME'] = os.getenv('PREFERRED_URL_SCHEME', 'http')

# Remove trailing slash if present
raw_server_name = raw_server_name.rstrip('/')

app.config['SERVER_NAME'] = raw_server_name

UPLOAD_FOLDER = Path(os.getenv('UPLOAD_FOLDER', 'uploads'))
OUTPUT_FOLDER = Path(os.getenv('OUTPUT_FOLDER', 'output'))
try:
    max_mb = int(os.getenv('MAX_UPLOAD_MB', '100'))
    if max_mb <= 0: max_mb = 100
except ValueError:
    max_mb = 100

try:
    mistral_max_mb = int(os.getenv('MISTRAL_MAX_MB', '50'))
    if mistral_max_mb <= 0: mistral_max_mb = 50
except ValueError:
    mistral_max_mb = 50

app.config['MAX_CONTENT_LENGTH'] = max_mb * 1024 * 1024
print(f"Maximum upload size set to: {max_mb} MB")
print(f"Mistral API max file size set to: {mistral_max_mb} MB")

ALLOWED_EXTENSIONS = {'pdf'}

# Directories are now created in the Dockerfile, no need to create them here.
# UPLOAD_FOLDER.mkdir(exist_ok=True)
# OUTPUT_FOLDER.mkdir(exist_ok=True)

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def split_pdf(pdf_path: Path, max_size_mb: int) -> list[Path]:
    """
    Splits a PDF into multiple parts, each approximately under max_size_mb.
    Returns list of split PDF paths.
    """
    print(f"Splitting PDF {pdf_path} into parts under {max_size_mb} MB...")
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    original_size = pdf_path.stat().st_size
    print(f"Original PDF has {total_pages} pages, size {original_size / (1024*1024):.2f} MB")

    # Estimate pages per split
    est_pages_per_split = max(1, int(total_pages * (max_size_mb * 1024 * 1024) / original_size))
    print(f"Estimated pages per split: {est_pages_per_split}")

    split_files = []
    part_num = 1
    page_start = 0

    while page_start < total_pages:
        writer = PdfWriter()
        page_end = min(page_start + est_pages_per_split, total_pages)
        for i in range(page_start, page_end):
            writer.add_page(reader.pages[i])

        split_path = pdf_path.parent / f"{pdf_path.stem}_part{part_num}.pdf"
        with open(split_path, "wb") as f_out:
            writer.write(f_out)

        split_size = split_path.stat().st_size / (1024*1024)
        print(f"  Created split {split_path.name} with pages {page_start+1}-{page_end}, size {split_size:.2f} MB")
        if split_size > max_size_mb:
            print(f"  Warning: split {split_path.name} exceeds {max_size_mb} MB limit ({split_size:.2f} MB)")

        split_files.append(split_path)
        part_num += 1
        page_start = page_end

    print(f"Splitting complete: {len(split_files)} parts created.")
    return split_files

# --- Core Processing Logic ---

def process_pdf(pdf_path: Path, api_key: str, session_output_dir: Path) -> tuple[str, str, list[str], Path, Path]:
    """
    Processes a single PDF file using Mistral OCR and saves results.

    Returns:
        A tuple (pdf_base_name, final_markdown_content, list_of_image_filenames, path_to_markdown_file, path_to_images_dir)
    Raises:
        Exception: For processing errors.
    """
    pdf_base = pdf_path.stem
    pdf_base_sanitized = secure_filename(pdf_base) # Use sanitized version for directory/file names
    print(f"Processing {pdf_path.name}...")

    pdf_output_dir = session_output_dir / pdf_base_sanitized # e.g., output/session_id/my_document/
    pdf_output_dir.mkdir(exist_ok=True)
    # Images will be saved directly in pdf_output_dir now

    client = Mistral(api_key=api_key)
    ocr_response: OCRResponse | None = None
    uploaded_file_id = None # Store only the ID for cleanup

    try:
        print(f"  Uploading {pdf_path.name} (size: {pdf_path.stat().st_size / (1024*1024):.2f} MB) to Mistral...")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        mistral_file = client.files.upload(
            file={"file_name": pdf_path.name, "content": pdf_bytes}, purpose="ocr"
        )
        uploaded_file_id = mistral_file.id # Store ID for cleanup

        print(f"  File uploaded (ID: {uploaded_file_id}). Getting signed URL...")
        signed_url = client.files.get_signed_url(file_id=uploaded_file_id, expiry=60) # Keep expiry short

        print(f"  Calling Mistral OCR API for file ID {uploaded_file_id} (this may take a while)...")
        start_time = time.time()
        ocr_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest", # Consider making model configurable if needed
            include_image_base64=True # Keep this True for image extraction
        )
        end_time = time.time()
        print(f"  OCR processing complete for {pdf_path.name} in {end_time - start_time:.2f} seconds.")

        # Save Raw OCR Response (Mandatory now)
        ocr_json_path = pdf_output_dir / "ocr_response.json"
        try:
            with open(ocr_json_path, "w", encoding="utf-8") as json_file:
                if hasattr(ocr_response, 'model_dump'):
                    json.dump(ocr_response.model_dump(), json_file, indent=4, ensure_ascii=False)
                else:
                     json.dump(ocr_response.dict(), json_file, indent=4, ensure_ascii=False)
            print(f"  Raw OCR response saved to {ocr_json_path}")
        except Exception as json_err:
            # Make saving JSON mandatory - raise error if it fails
            raise Exception(f"Failed to save raw OCR JSON response: {json_err}") from json_err

        # Process OCR Response -> Markdown & Images
        updated_markdown_pages = []
        extracted_image_filenames = [] # Store filenames for preview (original IDs, sanitized)
        image_path_updates = {} # Store mapping from original ID to relative path for markdown update

        print(f"  Extracting images and generating Markdown...")
        image_counter = 1
        for page_index, page in enumerate(ocr_response.pages):
            current_page_markdown = page.markdown # Start with original markdown

            for image_obj in page.images:
                if not image_obj.id or not image_obj.image_base64:
                    print(f"  Warning: Skipping image on page {page_index+1} due to missing ID or data.")
                    continue

                base64_str = image_obj.image_base64

                # Decode Base64
                if base64_str.startswith("data:"):
                    try:
                        base64_str = base64_str.split(",", 1)[1]
                    except IndexError:
                        print(f"  Warning: Malformed data URI for image on page {page_index+1}.")
                        continue
                try:
                    image_bytes = base64.b64decode(base64_str)
                except Exception as decode_err:
                    print(f"  Warning: Base64 decode error on page {page_index+1}: {decode_err}")
                    continue

                # Convert to webp using Pillow
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    webp_bytes_io = io.BytesIO()
                    image.save(webp_bytes_io, format="WEBP")
                    webp_bytes = webp_bytes_io.getvalue()
                except Exception as pil_err:
                    print(f"  Warning: Failed to convert image to webp on page {page_index+1}: {pil_err}")
                    continue

                # New filename: img-{counter}.webp
                new_img_name = f"img-{image_counter}.webp"
                image_output_path = pdf_output_dir / new_img_name

                # Save webp image
                try:
                    with open(image_output_path, "wb") as img_file:
                        img_file.write(webp_bytes)
                    extracted_image_filenames.append(new_img_name)
                    image_path_updates[image_obj.id] = new_img_name
                    print(f"    Saved image: {new_img_name}")
                except IOError as io_err:
                    print(f"  Warning: Could not write image file {image_output_path}: {io_err}")
                    continue

                image_counter += 1

            # Update markdown image references for the current page
            # Replace occurrences of ![alt](original_id) with ![Image X](new_filename)
            for original_id, new_filename in image_path_updates.items():
                # Replace alt text and filename
                # Pattern: ![any alt text](original_id) -> ![Image X](new_filename)
                current_page_markdown = current_page_markdown.replace(
                    f"]({original_id})", f"](/{new_filename})"
                )
                # Replace alt text before the bracket
                # This is a simple approach assuming standard markdown syntax
                current_page_markdown = current_page_markdown.replace(
                    f"![](/" + new_filename + ")", f"![Image {image_counter - 1}]({new_filename})"
                )

            updated_markdown_pages.append(current_page_markdown)
            image_path_updates = {} # Reset for next page

        final_markdown_content = "\n\n---\n\n".join(updated_markdown_pages) # Page separator
        # Change Markdown filename to just the base name + .md
        output_markdown_path = pdf_output_dir / f"{pdf_base_sanitized}.md"

        try:
            with open(output_markdown_path, "w", encoding="utf-8") as md_file:
                md_file.write(final_markdown_content)
            print(f"  Markdown generated successfully at {output_markdown_path}")
        except IOError as io_err:
            raise Exception(f"Failed to write final markdown file: {io_err}") from io_err

        # Return details needed for ZIP creation and preview
        # Note: images_dir is now just pdf_output_dir
        return pdf_base_sanitized, final_markdown_content, extracted_image_filenames, output_markdown_path, pdf_output_dir

    except Exception as e:
        error_str = str(e)
        # Attempt to extract JSON error message from the exception string
        json_index = error_str.find('{')
        if json_index != -1:
            try:
                error_json = json.loads(error_str[json_index:])
                error_msg = error_json.get("message", error_str)
            except Exception:
                error_msg = error_str
        else:
            error_msg = error_str # Use the raw error string if no JSON found
        print(f"  Error processing {pdf_path.name}: {error_msg}")
        # No need for separate cleanup here, finally block handles it
        raise Exception(error_msg) # Re-raise the simplified error message
    finally:
        # Ensure the uploaded file is deleted from Mistral even if errors occur
        if uploaded_file_id:
            try:
                print(f"  Attempting to delete temporary file {uploaded_file_id} from Mistral...")
                client.files.delete(file_id=uploaded_file_id)
                print(f"  Deleted temporary file {uploaded_file_id} from Mistral.")
            except Exception as delete_err:
                # Log warning but don't let cleanup failure hide the original error
                print(f"  Warning: Could not delete file {uploaded_file_id} from Mistral: {delete_err}")


def create_zip_archive(source_dir: Path, output_zip_path: Path):
    print(f"  Creating ZIP archive: {output_zip_path} from {source_dir}")
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for entry in source_dir.rglob('*'):
                arcname = entry.relative_to(source_dir)
                # Ensure arcname contains the 'images' folder if it exists
                # Example: arcname should be 'images/my_image.png', not just 'my_image.png'
                zipf.write(entry, arcname)
        print(f"  Successfully created ZIP: {output_zip_path}")
    except Exception as e:
        print(f"  Error creating ZIP file {output_zip_path}: {e}")
        raise


# --- Global job store & SSE Queues ---
jobs = {}
job_queues = {} # Stores message queues for SSE updates per job_id
jobs_lock = threading.Lock()

def _publish_status(job_id, status, data=None):
    """Helper to update job status and push to SSE queue."""
    with jobs_lock:
        if job_id not in jobs: return # Job might have been cancelled or timed out
        jobs[job_id]['status'] = status
        if data:
            jobs[job_id].update(data)

        # Push update to the SSE queue for this job
        if job_id in job_queues:
            update_message = {"status": status}
            if data:
                update_message.update(data)
            job_queues[job_id].put(json.dumps(update_message)) # Send JSON string

# --- Background worker function ---
def background_process_job(job_id, temp_pdf_path, api_key_to_use, session_id):
    _publish_status(job_id, 'processing')
    try:
        session_upload_dir = UPLOAD_FOLDER / session_id
        session_output_dir = OUTPUT_FOLDER / session_id
        session_output_dir.mkdir(parents=True, exist_ok=True)

        original_filename = temp_pdf_path.name
        pdf_base_sanitized = secure_filename(temp_pdf_path.stem)
        zip_filename = f"{pdf_base_sanitized}_output.zip"
        zip_output_path = session_output_dir / zip_filename
        final_output_dir = session_output_dir / pdf_base_sanitized
        final_output_dir.mkdir(exist_ok=True)

        # Check if file exceeds Mistral API limit
        if temp_pdf_path.stat().st_size > mistral_max_mb * 1024 * 1024:
            print(f"File exceeds {mistral_max_mb}MB, splitting before processing...")
            split_files = split_pdf(temp_pdf_path, mistral_max_mb)
            all_markdowns = []
            all_images = []
            image_counter = 1
            image_rename_map = {}

            for idx, split_file in enumerate(split_files, 1):
                print(f"Processing split part {idx}: {split_file.name}")
                part_base, md_content, img_files, md_path, img_dir = process_pdf(
                    split_file, api_key_to_use, session_output_dir
                )
                # Adjust image references in markdown
                for img_file in img_files:
                    old_path = img_dir / img_file
                    new_img_name = f"{image_counter}{Path(img_file).suffix}"
                    new_path = final_output_dir / new_img_name
                    shutil.copy2(old_path, new_path)
                    image_rename_map[img_file] = new_img_name
                    image_counter += 1

                # Update markdown image links
                for old_name, new_name in image_rename_map.items():
                    md_content = md_content.replace(f"({old_name})", f"({new_name})")

                all_markdowns.append(md_content)

                # Cleanup split part folder (optional)
                try:
                    split_folder = img_dir
                    if split_folder.exists() and split_folder != final_output_dir:
                        shutil.rmtree(split_folder)
                except Exception as e:
                    print(f"Warning: could not clean up split folder {split_folder}: {e}")

            merged_markdown = "\n\n---\n\n".join(all_markdowns)
            merged_md_path = final_output_dir / f"{pdf_base_sanitized}.md"
            with open(merged_md_path, "w", encoding="utf-8") as f_md:
                f_md.write(merged_markdown)
            print(f"Merged markdown saved to {merged_md_path}")

        else:
            # Process normally
            processed_pdf_base, markdown_content, image_filenames, md_path, img_dir = process_pdf(
                temp_pdf_path, api_key_to_use, session_output_dir
            )
            # Move markdown and images into final_output_dir if not already there
            if img_dir != final_output_dir:
                for img_file in image_filenames:
                    src = img_dir / img_file
                    dst = final_output_dir / img_file
                    shutil.copy2(src, dst)
                shutil.copy2(md_path, final_output_dir / md_path.name)

            # Delete the raw OCR JSON response before zipping
            ocr_json_path = img_dir / "ocr_response.json"
            try:
                if ocr_json_path.is_file():
                    ocr_json_path.unlink()
                    print(f"  Deleted raw OCR JSON file: {ocr_json_path}")
            except Exception as del_err:
                print(f"  Warning: Could not delete OCR JSON file {ocr_json_path}: {del_err}")

        # Delete the raw OCR JSON response in final_output_dir if exists
        final_ocr_json = final_output_dir / "ocr_response.json"
        try:
            if final_ocr_json.is_file():
                final_ocr_json.unlink()
                print(f"  Deleted raw OCR JSON file: {final_ocr_json}")
        except Exception as del_err:
            print(f"  Warning: Could not delete OCR JSON file {final_ocr_json}: {del_err}")

        # Create ZIP of the final output folder
        create_zip_archive(final_output_dir, zip_output_path)

        # Cleanup upload dir
        try:
            if session_upload_dir.exists():
                shutil.rmtree(session_upload_dir)
                print(f"  Cleaned up upload directory: {session_upload_dir}")
        except Exception as cleanup_err:
            print(f"  Warning: Could not cleanup upload directory {session_upload_dir}: {cleanup_err}")

        # Generate download URL
        download_url = None
        try:
            with app.app_context():
                download_url = url_for('download_file', session_id=session_id, filename=zip_filename, _external=True)
        except RuntimeError as url_err:
            print(f"  Error generating download URL: {url_err}")
            _publish_status(job_id, 'error', {"error": f"Processing complete, but failed to generate download link: {url_err}. Please configure SERVER_NAME."})
            return

        _publish_status(job_id, 'done', {"download_url": download_url})

    except Exception as e:
        _publish_status(job_id, 'error', {"error": str(e)})
    finally:
        # Signal end of stream by putting None or a specific marker
        if job_id in job_queues:
            job_queues[job_id].put(None) # Signal completion to the SSE generator

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html', max_upload_mb=max_mb, mistral_max_mb=mistral_max_mb)

@app.route('/process', methods=['POST'])
def handle_process():
    if 'pdf_files' not in request.files:
        return jsonify({"error": "No PDF files part in the request"}), 400

    files = request.files.getlist('pdf_files')
    env_api_key = os.getenv("MISTRAL_API_KEY")
    form_api_key = request.form.get('api_key')

    api_key_to_use = None
    if env_api_key:
        api_key_to_use = env_api_key
        print("Using API key from environment variable.")
    elif form_api_key:
        api_key_to_use = form_api_key
        print("Using API key from form input as fallback.")
    else:
        return jsonify({"error": "Mistral API Key is required. Set the MISTRAL_API_KEY environment variable or provide it in the form."}), 400

    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected PDF files"}), 400

    # Handle only the first valid PDF file asynchronously
    processed_file = False
    for f in files:
        if f and allowed_file(f.filename):
            job_id = str(uuid4())
            session_id = str(uuid4()) # Generate session ID here

            # Initialize job status and SSE queue
            with jobs_lock:
                jobs[job_id] = {"status": "queued", "download_url": None, "error": None, "session_id": session_id}
                job_queues[job_id] = queue.Queue() # Create a queue for this job's updates

            # Save uploaded file immediately
            try:
                session_upload_dir = UPLOAD_FOLDER / session_id
                session_upload_dir.mkdir(parents=True, exist_ok=True)
                filename_sanitized = secure_filename(f.filename)
                temp_pdf_path = session_upload_dir / filename_sanitized
                f.save(temp_pdf_path)
            except Exception as save_err:
                 with jobs_lock: # Clean up job entry if save fails
                     if job_id in jobs: del jobs[job_id]
                     if job_id in job_queues: del job_queues[job_id]
                 return jsonify({"error": f"Failed to save uploaded file: {save_err}"}), 500

            # Publish initial 'queued' status via SSE immediately after setup
            _publish_status(job_id, 'queued')

            # Start background thread
            t = threading.Thread(target=background_process_job, args=(job_id, temp_pdf_path, api_key_to_use, session_id))
            t.daemon = True # Allow app to exit even if threads are running (optional)
            t.start()
            processed_file = True
            # Redirect to the job page, which will connect to SSE
            return redirect(url_for('job_status', job_id=job_id))

    if not processed_file:
        return jsonify({"error": "No valid PDF files found or failed to process."}), 400

# Removed old /status/<job_id> route

# --- SSE Stream Route ---
@app.route('/stream/<job_id>')
def stream(job_id):
    def event_stream():
        q = job_queues.get(job_id)
        if not q:
            # Maybe the job finished very quickly or ID is invalid
            with jobs_lock:
                job = jobs.get(job_id)
                if job: # If job exists, send its final state
                    yield f"data: {json.dumps(job)}\n\n"
                else: # Job doesn't exist
                    yield f"data: {json.dumps({'status': 'error', 'error': 'Invalid or expired job ID'})}\n\n"
            return # End stream

        # Set timeout slightly longer than Gunicorn worker timeout (300s)
        sse_timeout = 310
        try:
            while True:
                message = q.get(timeout=sse_timeout) # Increased timeout
                if message is None: # End of stream signal from worker
                    break
                yield f"data: {message}\n\n" # SSE format: data: <json_string>\n\n
        except queue.Empty:
            # Timeout occurred, client might have disconnected or worker is stuck
            print(f"SSE stream timeout ({sse_timeout}s) for job {job_id}")
        finally:
            # Clean up the queue for this job_id when the client disconnects or stream ends
            with jobs_lock:
                if job_id in job_queues:
                    del job_queues[job_id]
                    print(f"Cleaned up SSE queue for job {job_id}")
                # Optionally clean up the job entry itself after a delay?
                # For now, keep job entry for potential later access via job_status page reload

    return Response(event_stream(), mimetype="text/event-stream")


# --- Route for serving images ---
@app.route('/view_image/<session_id>/<pdf_base>/<filename>')
def view_image(session_id, pdf_base, filename):
    """Serves an extracted image file for inline display."""
    safe_session_id = secure_filename(session_id)
    safe_pdf_base = secure_filename(pdf_base) # The folder name for the specific PDF
    safe_filename = secure_filename(filename) # The image filename (original sanitized ID)

    # Images are now directly inside the pdf_base folder
    directory = OUTPUT_FOLDER / safe_session_id / safe_pdf_base
    file_path = directory / safe_filename

    # Security check: Ensure the resolved path is still within the intended directory
    if not str(file_path.resolve()).startswith(str(directory.resolve())):
         return "Invalid path", 400
    if not file_path.is_file():
         return "Image not found", 404

    print(f"Serving image: {file_path}")
    # Send *without* as_attachment=True for inline display
    return send_from_directory(directory, safe_filename)


@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    """Serves the generated ZIP file for download."""
    safe_session_id = secure_filename(session_id)
    safe_filename = secure_filename(filename)
    directory = OUTPUT_FOLDER / safe_session_id
    file_path = directory / safe_filename

    if not str(file_path.resolve()).startswith(str(directory.resolve())): return "Invalid path", 400
    if not file_path.is_file(): return "File not found", 404

    print(f"Serving ZIP for download: {file_path}")
    return send_from_directory(directory, safe_filename, as_attachment=True)

@app.route('/job/<job_id>')
def job_status(job_id):
    # Render the basic page; SSE will provide dynamic updates.
    # We can pass the initial status if available, but JS will handle updates.
    with jobs_lock:
        job = jobs.get(job_id)
        initial_status = job['status'] if job else 'loading' # Or 'unknown'

    # Check if job exists, otherwise show error page immediately
    if not job:
         return render_template('job.html', job_id=job_id, status='error', error='Invalid or expired job ID')

    return render_template('job.html', job_id=job_id, status=initial_status) # Pass initial status

if __name__ == '__main__':
    # Set SERVER_NAME for external URL generation if running locally without Gunicorn/proxy
    # For production, this should ideally be handled by the proxy (e.g., Nginx)
    # or set via environment variables.
    # SERVER_NAME is now configured above

    host = os.getenv('FLASK_HOST', '127.0.0.1') # Gunicorn/Docker typically use 0.0.0.0
    port = FLASK_PORT_INT # Use the integer port loaded earlier
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(host=host, port=port, debug=debug_mode)
