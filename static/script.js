document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('pdf-files');
    const statusLog = document.getElementById('status-log');
    const loader = document.getElementById('loader');
    const resultsArea = document.getElementById('results-area');
    const downloadLinksList = document.getElementById('download-links');
    const errorArea = document.getElementById('error-area');
    const errorMessage = document.getElementById('error-message');

    function logStatus(message) {
        console.log(message);
        statusLog.textContent += message + '\n';
        statusLog.scrollTop = statusLog.scrollHeight;
    }

    function resetUI() {
        statusLog.textContent = '';
        downloadLinksList.innerHTML = '';
        resultsArea.style.display = 'none';
        errorArea.style.display = 'none';
        errorMessage.textContent = '';
        loader.style.display = 'none';
    }

    async function uploadFiles(files) {
        resetUI();
        if (!files.length) {
            logStatus('No files selected.');
            return;
        }

        logStatus(`Uploading ${files.length} file(s)...`);
        loader.style.display = 'block';

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('pdf_files', files[i]);
            logStatus(` - Added file: ${files[i].name} (${(files[i].size / (1024*1024)).toFixed(2)} MB)`);
        }

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData,
            });

            logStatus(`Server response: ${response.status} ${response.statusText}`);

            if (!response.ok) {
                let errorData = { error: `Server error: ${response.status} ${response.statusText}` };
                try {
                    errorData = await response.json();
                } catch (e) {}
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            const result = await response.json();
            if (!result.job_id) {
                throw new Error(result.error || 'Unexpected response: missing job_id');
            }

            const jobId = result.job_id;
            logStatus(`Job started. ID: ${jobId}`);
            pollStatus(jobId);

        } catch (error) {
            logStatus(`Error during upload: ${error.message}`);
            errorMessage.textContent = `Error: ${error.message}`;
            errorArea.style.display = 'block';
            loader.style.display = 'none';
        }
    }

    function addDownloadLink(url) {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = url;
        a.textContent = 'Download ZIP';
        a.target = '_blank';
        li.appendChild(a);
        downloadLinksList.appendChild(li);
        resultsArea.style.display = 'block';
    }

    function pollStatus(jobId) {
        const pollIntervalMs = 3000;
        const maxPollTimeMs = 10 * 60 * 1000; // 10 minutes
        const startTime = Date.now();

        async function poll() {
            try {
                const resp = await fetch(`/status/${jobId}`);
                if (!resp.ok) throw new Error(`Status check failed: ${resp.status}`);
                const data = await resp.json();
                logStatus(`Job ${jobId} status: ${data.status}`);

                if (data.status === 'done' && data.download_url) {
                    logStatus('Processing complete. Download ready.');
                    addDownloadLink(data.download_url);

                    // Auto-trigger download
                    const a = document.createElement('a');
                    a.href = data.download_url;
                    a.download = '';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);

                    loader.style.display = 'none';
                    return;
                } else if (data.status === 'error') {
                    throw new Error(data.error || 'Unknown error during processing');
                } else if (Date.now() - startTime > maxPollTimeMs) {
                    throw new Error('Processing timed out.');
                } else {
                    setTimeout(poll, pollIntervalMs);
                }
            } catch (err) {
                logStatus(`Error: ${err.message}`);
                errorMessage.textContent = `Error: ${err.message}`;
                errorArea.style.display = 'block';
                loader.style.display = 'none';
            }
        }

        poll();
    }

    // Drag and drop handlers
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        uploadFiles(files);
    });

    fileInput.addEventListener('change', () => {
        uploadFiles(fileInput.files);
    });
});
