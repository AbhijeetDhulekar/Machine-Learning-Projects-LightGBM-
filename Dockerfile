FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


RUN ls -l templates/ && \
    ls -l static/ && \
    [ -f templates/titanic_prediction.html ] && \
    [ -f static/logo.png ]


# Download model files (if not included in repo)
# ADD https://your-model-storage/lgbm_model.joblib ./lgbm_model.joblib
# ADD https://your-model-storage/scaler.joblib ./scaler.joblib

# Expose port
EXPOSE 5005

CMD ["gunicorn", "--bind", "0.0.0.0:5005", "--workers", "4", "app:app"]
