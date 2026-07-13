# Example Dockerfile for TF executor.
# Base image version is an example — actual image configured via docker.images.
FROM tensorflow/tensorflow:2.15.0-gpu
# Install executor deps (tf base already has tensorflow; add per-op deps here)
RUN pip install numpy pyyaml
COPY . /executor/
ENTRYPOINT ["python", "/executor/executor_main.py"]
