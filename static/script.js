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
        logStatus("Form submitted.");
        resetUI();

        const apiKey = apiKeyInput.value.trim();
        const files = fileInput.files;
        logStatus(`API Key provided: ${apiKey ? 'Yes' : 'No (using environment variable if set)'}`);
        logStatus(`Number of files selected: ${files.length}`);

        if (files.length === 0) {
            logStatus('Error: At least one PDF file is required.');
            errorMessage.textContent = 'At least one PDF file is required.';
            errorArea.style.display = 'block';
            return;
        }

        submitBtn.disabled = true;
        loader.style.display = 'block';
        logStatus('Uploading PDF and starting background processing...');

        const formData = new FormData();
        formData.append('api_key', apiKey);
        for (let i = 0; i < files.length; i++) {
            formData.append('pdf_files', files[i]);
            logStatus(` - Added file to form data: ${files[i].name} (${(files[i].size / (1024*1024)).toFixed(2)} MB)`);
        }

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData,
            });
            logStatus(`Received response: ${response.status} ${response.statusText}`);

            if (!response.ok) {
                let errorData = { error: `Server error: ${response.status} ${response.statusText}` };
                try {
                    errorData = await response.json();
                } catch (e) {}
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            const result = await response.json();
            if (!result.job_id) {
                throw new Error('Unexpected response: missing job_id');
            }

            logStatus(`Job started with ID: ${result.job_id}`);
            logStatus('Polling for job status...');

            const pollIntervalMs = 3000;
            const maxPollTimeMs = 10 * 60 * 1000; // 10 minutes
            const startTime = Date.now();

            async function pollStatus() {
                try {
                    const statusResp = await fetch(`/status/${result.job_id}`);
                    if (!statusResp.ok) {
                        throw new Error(`Status check failed: ${statusResp.status}`);
                    }
                    const statusData = await statusResp.json();
                    logStatus(`Job status: ${statusData.status}`);

                    if (statusData.status === 'done' && statusData.download_url) {
                        logStatus('Processing complete! Downloading ZIP...');
                        // Auto-trigger download
                        const a = document.createElement('a');
                        a.href = statusData.download_url;
                        a.download = '';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        logStatus('Download started.');
                        submitBtn.disabled = false;
                        loader.style.display = 'none';
                        return;
                    } else if (statusData.status === 'error') {
                        throw new Error(statusData.error || 'Unknown error during processing');
                    } else if (Date.now() - startTime > maxPollTimeMs) {
                        throw new Error('Processing timed out.');
                    } else {
                        setTimeout(pollStatus, pollIntervalMs);
                    }
                } catch (err) {
                    logStatus(`>>> Error: ${err.message}`);
                    errorMessage.textContent = `Error: ${err.message}`;
                    errorArea.style.display = 'block';
                    submitBtn.disabled = false;
                    loader.style.display = 'none';
                }
            }

            pollStatus();

        } catch (error) {
            logStatus(`>>> Error during upload: ${error.message}`);
            console.error('Caught error:', error);
            errorMessage.textContent = `Error: ${error.message}`;
            errorArea.style.display = 'block';
            submitBtn.disabled = false;
            loader.style.display = 'none';
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
