# Mistral PDF-to-Markdown OCR 📄➡️📝

[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-WebApp-green?logo=flask)](https://flask.palletsprojects.com/)

A simple web application using Flask and the Mistral AI API to convert PDF files into standard Markdown documents, extracting text and images. Easily run using Docker Compose.

*(Inspired by an Obsidian version at [diegomarzaa/pdf-ocr-obsidian](https://github.com/diegomarzaa/pdf-ocr-obsidian))*

## ✨ Features

-   **⬆️ Upload Interface:** Simple web UI for uploading PDFs.
-   **🤖 Mistral OCR:** Leverages [Mistral OCR](https://mistral.ai/news/mistral-ocr) for accurate text/image extraction.
-   **📄 Standard Markdown:** Outputs clean Markdown with relative image links.
-   **🖼️ Image Handling:** Saves extracted images alongside the Markdown file using original filenames.
-   **📦 Packaged Output:** Delivers results (Markdown, images, raw JSON response) as a downloadable ZIP archive.
-   **⚙️ Configurable Upload Limit:** Set maximum PDF upload size via environment variable (default 100MB).
-   **🐳 Dockerized:** Ready to run with Docker Compose.

## 🚀 Getting Started

### Prerequisites

-   [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/) installed.
-   A Mistral AI API Key from [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys).

### Running the App

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create `.env` File:**
    Create a file named `.env` in the project root and add your API key:
    ```dotenv
    # .env
    MISTRAL_API_KEY=your_actual_api_key_here
    ```
    *(This file is git-ignored, keeping your key safe!)*

3.  **Launch with Docker Compose:**
    ```bash
    docker-compose up --build -d
    ```
    *   `--build`: Rebuilds the image if needed.
    *   `-d`: Runs the container in the background (detached mode).

4.  **Access the Web App:**
    Open your browser and go to: `http://localhost:5009` 🎉

5.  **Stopping the App:**
    ```bash
    docker-compose down
    ```

## ⚙️ How It Works

1.  Upload PDFs via the web UI.
2.  The Flask backend sends files to the Mistral OCR API.
3.  Mistral processes the PDF, returning text and images.
4.  The app saves the text as `.md` and images in an `images/` subfolder.
5.  Everything is zipped up for you to download.

## 📜 License

This project is licensed under the Mozilla Public License Version 2.0. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
