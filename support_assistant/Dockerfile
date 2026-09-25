
FROM python:3.11-slim
WORKDIR /app
COPY support_assistant /app/support_assistant
RUN pip install --no-cache-dir -r /app/support_assistant/requirements.txt
WORKDIR /app
RUN python -m support_assistant.ingestion
CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]
