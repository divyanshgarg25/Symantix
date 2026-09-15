# Builder Stage
FROM python:3.11-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Final Stage
FROM python:3.11-slim

WORKDIR /app
# Add non-root user for security
RUN groupadd -r symantix && useradd -r -g symantix symantixuser

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /wheels/*

# Copy backend code
COPY backend/ ./backend/
# Copy core-ml weights if any
COPY core-ml/ ./core-ml/

RUN chown -R symantixuser:symantix /app
USER symantixuser

EXPOSE 8080

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8080"]
