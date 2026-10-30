FROM python:3.11-slim

WORKDIR /job

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY drift_check.py .

ENV PREDICTIONS_LOG=/logs/predictions.jsonl
ENV REFERENCE_CSV=/data/reference.csv
ENV DRIFT_REPORT_HTML=/reports/drift_report.html
ENV DRIFT_INTERVAL_SEC=120
ENV DRIFT_METRICS_PORT=9100

CMD ["python", "drift_check.py"]
