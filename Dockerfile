FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY machine_learning/requirements.txt ./ml_req.txt
COPY system/requirements.txt ./sys_req.txt
RUN pip install --no-cache-dir -r ml_req.txt && pip install --no-cache-dir -r sys_req.txt

COPY . .
CMD ["tail", "-f", "/dev/null"]