# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files to container
COPY absenteeism_model.pkl /app/
COPY serve_api.py /app/
COPY params.yaml /app/
COPY /data/raw/Absenteeism_at_work.csv /app/data/raw/

# Install dependencies
RUN pip install pandas fastapi uvicorn scikit-learn pydantic 
RUN pip install pyyaml

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8000"]