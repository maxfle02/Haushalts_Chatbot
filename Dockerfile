# Base Image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y git && apt-get clean

# Copy the ChromaDB directory if it exists
COPY chroma_db /app/chroma_db

# Copy the rest of the application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Add a shell script to run test.py and then app.py
RUN echo '#!/bin/sh\n\
if [ ! -d /app/chroma_db ]; then\n\
  echo "Error: /app/chroma_db not found!"; exit 1;\n\
fi\n\
python src/test.py && streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0' > /app/start.sh && chmod +x /app/start.sh

# Run the shell script
CMD ["/app/start.sh"]
