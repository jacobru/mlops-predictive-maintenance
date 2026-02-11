# 1. Use an official Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first (for faster building)
COPY requirements.txt .

# 4. Install the libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your code and the trained model
COPY src/ ./src/
COPY models/ ./models/

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to run the API when the container starts
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]