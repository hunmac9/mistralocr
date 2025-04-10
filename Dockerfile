# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure Python output is sent straight to terminal without buffering
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for certain Python packages)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# Ensure permissions are set correctly for the non-root user later
COPY --chown=appuser:appgroup . .

# Make port 5009 available to the world outside this container
# This is the default port used in app.py
EXPOSE 5009

# Define environment variables for Flask/Gunicorn (optional, can be overridden)
# Gunicorn will bind to 0.0.0.0:5009 based on the CMD below
# ENV FLASK_APP=app.py # Not strictly needed when using gunicorn app:app format
# ENV FLASK_RUN_HOST=0.0.0.0 # Used by flask run, gunicorn uses --bind
# ENV FLASK_RUN_PORT=5009 # Used by flask run, gunicorn uses --bind

# Define the command to run the application using Gunicorn
# Bind to 0.0.0.0 to allow external connections to the container
# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Switch to the non-root user
USER appuser

# Define the command to run the application using Gunicorn
# Bind to 0.0.0.0 to allow external connections to the container
# Use the default port 5009
CMD ["gunicorn", "--bind", "0.0.0.0:5009", "app:app"]
