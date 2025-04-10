/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('ocr-form');
    const apiKeyInput = document.getElementById('api-key');
    const fileInput = document.getElementById('pdf-files');
    const apiKeyToggle = document.getElementById('api-key-toggle'); 
    const submitBtn = document.getElementById('submit-btn');
    const statusLog = document.getElementById('status-log');
    const loader = document.getElementById('loader');

    // Result Areas
    const resultsArea = document.getElementById('results-area');
    const downloadLinksList = document.getElementById('download-links');
    const previewArea = document.getElementById('preview-area'); // New preview area
    const previewContent = document.getElementById('preview-content'); // Container for previews

    // Error Area
    const errorArea = document.getElementById('error-area');
    const errorMessage = document.getElementById('error-message');

    function logStatus(message) {
        console.log(message);
        statusLog.textContent += message + '\n';
        statusLog.scrollTop = statusLog.scrollHeight; // Auto-scroll
    }

    function resetUI() {
        logStatus("Resetting UI elements."); // Added log
        statusLog.textContent = '';
        resultsArea.style.display = 'none';
        downloadLinksList.innerHTML = '';
        previewArea.style.display = 'none';   // Hide preview area
        previewContent.innerHTML = '';        // Clear preview content
        errorArea.style.display = 'none';
        errorMessage.textContent = '';
        submitBtn.disabled = false;
        loader.style.display = 'none';
        logStatus("UI reset complete."); // Added log
    }

    // --- Function to render preview (Images Only) ---
    function renderPreview(resultItem, sessionId) {
        logStatus(`Attempting to render preview for: ${resultItem.original_filename}`); // Added log
        if (!resultItem.preview || !resultItem.preview.images || resultItem.preview.images.length === 0) {
             logStatus(` - No images found or preview data missing for ${resultItem.original_filename}. Skipping image preview.`); // Added log
             return; // Skip if no images
        }


        const previewContainer = document.createElement('div');
        previewContainer.classList.add('preview-item');
        previewContainer.classList.add('collapsed'); // Initially collapse

        const title = document.createElement('h3');
        title.textContent = `Image Preview for: ${resultItem.original_filename}`; // Updated title
        previewContainer.appendChild(title);

        // --- Create the TOGGLE element ---
        const toggleButton = document.createElement('div');
        toggleButton.classList.add('preview-toggle');
        toggleButton.textContent = 'Show/Hide Preview'; // Initial text (will be updated by CSS)
        previewContainer.appendChild(toggleButton);



        // --- Create the INNER CONTENT div (that will be shown/hidden) ---
        const contentInner = document.createElement('div');
        contentInner.classList.add('preview-content-inner');

        // --- REMOVED Markdown Preview Section ---


        // --- Image Preview ---
        // This section remains largely the same, just ensure it's directly under contentInner
        const imageSection = document.createElement('div');
        imageSection.classList.add('image-preview');
        const imageTitle = document.createElement('h4');
        imageTitle.textContent = 'Extracted Images'; // Simplified title
        imageSection.appendChild(imageTitle);

        logStatus(` - Rendering ${resultItem.preview.images.length} image(s) for ${resultItem.original_filename}`); // Added log
        resultItem.preview.images.forEach(imageFilename => {
            const img = document.createElement('img');
            const imageUrl = `/view_image/${sessionId}/${resultItem.preview.pdf_base}/${imageFilename}`;
            logStatus(`   - Adding image: ${imageUrl}`); // Log image URL
            img.src = imageUrl;
            img.alt = imageFilename; // Use the raw filename as alt text
                img.style.maxWidth = '150px';
                img.style.height = 'auto';
                img.style.margin = '5px';
                img.style.border = '1px solid #ddd';
                img.style.display = 'inline-block';
                img.onerror = () => {
                    logStatus(`   - Error loading image: ${imageUrl}`); // Log image load error
                    img.alt = `Could not load: ${imageFilename}`;
                    img.style.border = '1px solid red';
                 };
                imageSection.appendChild(img);
            });
        contentInner.appendChild(imageSection);
        // } // End of the original if check (now removed as the outer function checks)

        // --- Append the inner content to the preview container ---
        previewContainer.appendChild(contentInner);


        // --- Add EVENT LISTENER to toggle button ---
        toggleButton.addEventListener('click', () => {
            previewContainer.classList.toggle('collapsed'); // Toggle collapsed class
        });

        previewContent.appendChild(previewContainer);
        logStatus(` - Finished rendering preview for: ${resultItem.original_filename}`); // Added log
    }


    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        logStatus("Form submitted."); // Added log
        resetUI();

        const apiKey = apiKeyInput.value.trim(); // Still read the key, might be empty
        const files = fileInput.files;
        logStatus(`API Key provided: ${apiKey ? 'Yes' : 'No (using environment variable if set)'}`); // Log API key status
        logStatus(`Number of files selected: ${files.length}`); // Log file count

        // Only require files on the client-side. Server handles API key logic.
        if (files.length === 0) {
             logStatus('Error: At least one PDF file is required.');
             errorMessage.textContent = 'At least one PDF file is required.';
             errorArea.style.display = 'block';
             return;
        }

        submitBtn.disabled = true;
        loader.style.display = 'block';
        logStatus('Starting PDF processing...');

        const formData = new FormData();
        formData.append('api_key', apiKey);
        for (let i = 0; i < files.length; i++) {
            formData.append('pdf_files', files[i]);
            logStatus(` - Added file to form data: ${files[i].name} (Size: ${(files[i].size / (1024*1024)).toFixed(2)} MB)`); // More detailed log
        }

        try {
            logStatus('Sending request to /process endpoint...'); // Updated log
            const response = await fetch('/process', {
                method: 'POST',
                body: formData,
                // Consider adding signal for AbortController if needed for cancellation
            });
            logStatus(`Received response from server. Status: ${response.status} ${response.statusText}`); // Added log

            if (!response.ok) {
                let errorData = { error: `Server error: ${response.status} ${response.statusText}` };
                try {
                    logStatus("Attempting to parse error response JSON..."); // Added log
                    errorData = await response.json();
                    logStatus("Parsed error response JSON successfully."); // Added log
                } catch (e) {
                    logStatus("Could not parse error response as JSON."); // Added log
                    /* Ignore if response not JSON */
                }
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            logStatus("Parsing successful response JSON..."); // Added log
            const result = await response.json();
            logStatus("Parsed successful response JSON."); // Added log
            // console.log("Server Response:", result); // Optional: Detailed console log for debugging

            if (result.success && result.results && result.session_id) { // Check for session_id
                logStatus(`Processing successful! Session ID: ${result.session_id}`); // Updated log
                const sessionId = result.session_id; // Get session ID

                // Populate Download Links & Previews
                if (result.results.length > 0) {
                    logStatus(`Processing ${result.results.length} result item(s)...`); // Added log
                    resultsArea.style.display = 'block'; // Show downloads area
                    result.results.forEach((item, index) => {
                        logStatus(` - Processing result ${index + 1}: ${item.original_filename}`); // Log item processing
                        const li = document.createElement('li');
                        const link = document.createElement('a');
                        link.href = item.download_url;
                        link.textContent = `Download ${item.zip_filename}`;
                        link.setAttribute('download', item.zip_filename); // Suggest filename on download
                        li.appendChild(link);
                        downloadLinksList.appendChild(li);
                        logStatus(`   - Added download link for ${item.zip_filename}`); // Log link addition

                        // --- Generate Preview for this item ---
                        renderPreview(item, sessionId); // Function now handles logging inside
                    });

                    if (previewContent.hasChildNodes()) {
                       logStatus("Displaying preview area."); // Added log
                       previewArea.style.display = 'block'; // Show preview area if previews were added
                    } else {
                       logStatus("No previews generated (likely no images found)."); // Added log
                    }

                } else {
                     logStatus("Processing finished, but no successful results returned from server."); // Updated log
                }


                 // Display any partial errors/warnings
                 if (result.errors && result.errors.length > 0) {
                    logStatus('\n--- Warnings/Partial Errors ---');
                    result.errors.forEach(err => logStatus(`- ${err}`)); // Log each specific warning
                    logStatus('-------------------------------\n');
                } else {
                    logStatus("No warnings or partial errors reported by server."); // Added log
                }

            } else if (result.error) {
                 logStatus(`Server returned an error message: ${result.error}`); // Added log
                 throw new Error(result.error);
            } else {
                 logStatus("Server response was successful but structure was unexpected."); // Added log
                 throw new Error('Received unexpected response format from server.'); // More specific error
            }

        } catch (error) {
            logStatus(`>>> Error during processing: ${error.message}`); // Emphasize error log
            console.error('Caught processing error:', error); // Keep console error for details
            errorMessage.textContent = `Error: ${error.message}`; // Add "Error: " prefix
            errorArea.style.display = 'block';
        } finally {
            submitBtn.disabled = false;
            loader.style.display = 'none';
            logStatus('Processing finished. Ready for next operation.'); // Updated final log
        }
    });

    if (apiKeyToggle && apiKeyInput) {
        apiKeyToggle.addEventListener('change', function() {
            if (this.checked) {
                apiKeyInput.type = 'text';
            } else {
                apiKeyInput.type = 'password';
            }
        });
    }  
});
