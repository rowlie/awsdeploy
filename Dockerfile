# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install all dependencies including gunicorn
# We use a single RUN command to ensure all packages are installed together.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gunicorn && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app will run on
EXPOSE 8080

# Define environment variable for the port
ENV PORT 8080

# CRITICAL: The CMD line is REMOVED. 
# We rely entirely on the Start Command defined in the AWS console
# (gunicorn --bind 0.0.0.0:8080 ... app:app) to launch the service.
