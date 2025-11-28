# Use an official Python runtime as a parent image
# We use a stable version of Python for reliability.
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir saves space
# We also install 'gunicorn', a production-ready WSGI server, which is better than Flask's built-in server.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir gunicorn
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app will run on
# AWS App Runner typically uses the PORT environment variable, but 8080 is a good default.
EXPOSE 8080

# Define environment variable for the port (optional, but good practice)
ENV PORT 8080

# Run gunicorn to serve the Flask app
# The format is: gunicorn --bind 0.0.0.0:$PORT <app_file>:<app_instance>
# In your case: app.py contains the 'app = Flask(__name__)' instance.
# The --timeout 120 is important for LangChain/LLM calls that might take longer.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]