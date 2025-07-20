# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main script and contracts folder
COPY knowledge-graph.py .
COPY contracts/ ./contracts/

# Create output directory
RUN mkdir -p output

# Set environment variables
ENV GEMMA_URL=https://gemma-3b-1011220518354.europe-west1.run.app/
ENV GEMMA_MODEL=gemma3:1b

# Expose port (if you later add a web interface)
EXPOSE 8000

# Default command to run the script
CMD ["python", "knowledge-graph.py"]

# Alternative: Run in interactive mode
# CMD ["python", "-i", "knowledge-graph.py"] 