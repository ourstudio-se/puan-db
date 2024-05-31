# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Expose port 8000 for the app to be accessible
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "app.http_server:app", "--host", "0.0.0.0", "--port", "8000"]
