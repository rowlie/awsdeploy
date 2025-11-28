# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install all dependencies including gunicorn
# We use a single RUN command to install all packages at once
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gunicorn && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app will run on
EXPOSE 8080

# Define environment variable for the port
ENV PORT 8080

# --- CRITICAL FIX HERE: Using shell form (single string) for better PATH discovery ---
# This ensures gunicorn is found and executed correctly.
CMD gunicorn --bind 0.0.0.0:8080 --workers 2 --threads 4 --timeout 120 app:app
