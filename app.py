# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import json
import base64
import shutil
import zipfile
from pathlib import Path
from uuid import uuid4
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

app = Flask(__name__)

# --- Configuration ---
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
        # Check if file exceeds Mistral API limit
        if pdf_path.stat().st_size > mistral_max_mb * 1024 * 1024:
            print(f"  PDF exceeds {mistral_max_mb}MB. Compressing before upload...")
            compressed_path = pdf_path.parent / f"{pdf_path.stem}_compressed.pdf"
            # Use Ghostscript or similar to compress PDF
            import subprocess
            gs_cmd = [
                "gs",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.4",
                "-dPDFSETTINGS=/screen",
                "-dNOPAUSE",
                "-dQUIET",
                "-dBATCH",
                f"-sOutputFile={compressed_path}",
                str(pdf_path)
            ]
            try:
                subprocess.run(gs_cmd, check=True)
                if compressed_path.exists() and compressed_path.stat().st_size < pdf_path.stat().st_size:
                    print(f"  Compression successful. Using compressed file: {compressed_path.name}")
                    pdf_path = compressed_path
                else:
                    print("  Compression did not reduce size. Using original file.")
            except Exception as e:
                print(f"  Compression failed: {e}. Using original file.")

        print(f"  Uploading {pdf_path.name} to Mistral...")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        mistral_file = client.files.upload(
            file={"file_name": pdf_path.name, "content": pdf_bytes}, purpose="ocr"
        )
        uploaded_file_id = mistral_file.id # Store ID for cleanup

        print(f"  File uploaded (ID: {uploaded_file_id}). Getting signed URL...")
        signed_url = client.files.get_signed_url(file_id=uploaded_file_id, expiry=60)

        print(f"  Calling OCR API...")
        ocr_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )
        print(f"  OCR processing complete for {pdf_path.name}.")

        # Optional: Save Raw OCR Response
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
        for page_index, page in enumerate(ocr_response.pages):
            current_page_markdown = page.markdown # Start with original markdown

            for image_obj in page.images:
                if not image_obj.id or not image_obj.image_base64:
                    print(f"  Warning: Skipping image on page {page_index+1} due to missing ID or data.")
                    continue

                base64_str = image_obj.image_base64

                # Sanitize the original image ID provided by Mistral to use as filename
                # Keep the original extension if present, default to .png otherwise
                original_image_id = image_obj.id
                sanitized_image_filename = secure_filename(original_image_id)
                if not Path(sanitized_image_filename).suffix:
                     sanitized_image_filename += ".png" # Add default extension if missing

                # Path to save the image (directly in the pdf_output_dir)
                image_output_path = pdf_output_dir / sanitized_image_filename

                # Decode Base64
                if base64_str.startswith("data:"):
                    try: base64_str = base64_str.split(",", 1)[1]
                    except IndexError:
                         print(f"  Warning: Malformed data URI for image {original_image_id} on page {page_index+1}.")
                         continue
                try:
                    image_bytes = base64.b64decode(base64_str)
                except Exception as decode_err:
                    print(f"  Warning: Base64 decode error for image {original_image_id} on page {page_index+1}: {decode_err}")
                    continue

                # Save the image
                try:
                    with open(image_output_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    if sanitized_image_filename not in extracted_image_filenames: # Avoid duplicates if ID appears on multiple pages
                        extracted_image_filenames.append(sanitized_image_filename)
                    # Store mapping for potential markdown update (use original ID as key)
                    image_path_updates[original_image_id] = sanitized_image_filename
                    print(f"    Saved image: {sanitized_image_filename}")
                except IOError as io_err:
                     print(f"  Warning: Could not write image file {image_output_path}: {io_err}")
                     continue

            # Update markdown image references for the current page
            # Replace occurrences of ![alt](original_image_id) with ![alt](sanitized_image_filename)
            for original_id, new_filename in image_path_updates.items():
                 # Basic replacement - might need refinement for complex markdown structures
                 # This assumes Mistral uses the image ID directly in the markdown link like `![...](image_id)`
                 current_page_markdown = current_page_markdown.replace(f"({original_id})", f"({new_filename})")

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
        # Neither environment variable nor form input has the key
        return jsonify({"error": "Mistral API Key is required. Set the MISTRAL_API_KEY environment variable or provide it in the form."}), 400

    if not files or all(f.filename == '' for f in files):
         return jsonify({"error": "No selected PDF files"}), 400

    valid_files, invalid_files = [], []
    for f in files:
        if f and allowed_file(f.filename): valid_files.append(f)
        elif f and f.filename != '': invalid_files.append(f.filename)

    if not valid_files:
         # ... (error handling as before) ...
         error_msg = "No valid PDF files found."
         if invalid_files: error_msg += f" Invalid files skipped: {', '.join(invalid_files)}"
         return jsonify({"error": error_msg}), 400


    session_id = str(uuid4())
    session_upload_dir = UPLOAD_FOLDER / session_id
    session_output_dir = OUTPUT_FOLDER / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    session_output_dir.mkdir(parents=True, exist_ok=True)

    processed_files_results = [] # Changed name for clarity
    processing_errors = []
    if invalid_files: processing_errors.append(f"Skipped non-PDF files: {', '.join(invalid_files)}")

    for file in valid_files:
        original_filename = file.filename
        filename_sanitized = secure_filename(original_filename)
        pdf_base_sanitized = secure_filename(Path(original_filename).stem) # Get sanitized base name
        temp_pdf_path = session_upload_dir / filename_sanitized
        zip_filename = f"{pdf_base_sanitized}_output.zip"
        zip_output_path = session_output_dir / zip_filename
        individual_output_dir = session_output_dir / pdf_base_sanitized # Dir containing MD + images/

        try:
            print(f"Saving uploaded file temporarily to: {temp_pdf_path}")
            file.save(temp_pdf_path)

            # Process PDF - Capture new return values, passing the determined key
            processed_pdf_base, markdown_content, image_filenames, md_path, img_dir = process_pdf(
                temp_pdf_path, api_key_to_use, session_output_dir
            )

            # Create ZIP (using the individual output dir)
            create_zip_archive(individual_output_dir, zip_output_path)

            download_url = url_for('download_file', session_id=session_id, filename=zip_filename, _external=True)

            # Store results including preview data
            processed_files_results.append({
                "original_filename": original_filename, # Keep original name for display
                "zip_filename": zip_filename,
                "download_url": download_url,
                "preview": {
                    "markdown": markdown_content,
                    "images": image_filenames,
                    "pdf_base": processed_pdf_base # Use the sanitized base name returned by process_pdf
                }
            })
            print(f"Successfully processed and zipped: {original_filename}")

        except Exception as e:
            print(f"ERROR: Failed processing {original_filename}: {e}")
            processing_errors.append(f"{original_filename}: Processing Error - {e}")
        finally:
            if temp_pdf_path.exists():
                try: temp_pdf_path.unlink()
                except OSError as unlink_err: print(f"Warning: Could not delete temp file {temp_pdf_path}: {unlink_err}")

    # Cleanup session upload dir
    try:
        shutil.rmtree(session_upload_dir)
        print(f"Cleaned up session upload directory: {session_upload_dir}")
    except OSError as rmtree_err:
        print(f"Warning: Could not delete session upload directory {session_upload_dir}: {rmtree_err}")

    if not processed_files_results and processing_errors:
         return jsonify({"error": "All PDF processing attempts failed.", "details": processing_errors}), 500
    elif not processed_files_results:
         return jsonify({"error": "No files were processed successfully."}), 500
    else:
        # Return session_id along with results for constructing image URLs on frontend
        return jsonify({
            "success": True,
            "session_id": session_id,
            "results": processed_files_results, # Renamed from 'downloads'
            "errors": processing_errors
        }), 200

# --- NEW ROUTE for serving images ---
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

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5009))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(host=host, port=port, debug=debug_mode)
