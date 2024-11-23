# https://hub.docker.com/r/semitechnologies/transformers-inference/tags?name=sentence-transformers
# download specific images for either ARM64 or AMD64 (w/ CUDA support)
# ARM64: 0.69GB, AMD64: 5.97GB
# docker pull 
FROM semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L6-v2-1.9.7

WORKDIR /app

COPY src/requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY src/ /app
RUN python3 install.py

# Streamlit port expose and run
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]

# build instructions:
# docker build -t krirag -f krirag-onnx-multiplatform.dockerfile .
# run:
# docker run -p 8501:8501 krirag