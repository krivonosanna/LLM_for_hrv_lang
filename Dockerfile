FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

COPY model/ ./model/
COPY tokenizer/ ./tokenizer/

ENTRYPOINT ["python", "-m", "src.predict"]
