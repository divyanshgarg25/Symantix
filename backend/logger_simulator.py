import asyncio
import uuid
import random
import json
import aio_pika
from datetime import datetime, timezone
from typing import AsyncGenerator

from .models import LogEvent
from .config import settings

SERVICES = [
    "Frontend-Gateway",
    "Auth-Service",
    "Payment-API",
    "User-PostgreSQL"
]

async def generate_healthy_logs(trace_id: str) -> AsyncGenerator[LogEvent, None]:
    """Generates a standard, healthy request flow across microservices."""
    yield LogEvent(
        trace_id=trace_id,
        service_name="Frontend-Gateway",
        log_level="INFO",
        message="Received incoming HTTP GET request for /api/v1/profile",
        metadata={"path": "/api/v1/profile", "method": "GET"}
    )
    await asyncio.sleep(0.05)
    
    yield LogEvent(
        trace_id=trace_id,
        service_name="Auth-Service",
        log_level="INFO",
        message="Validating JWT token for user session.",
        metadata={"action": "jwt_validate"}
    )
    await asyncio.sleep(0.1)
    
    yield LogEvent(
        trace_id=trace_id,
        service_name="User-PostgreSQL",
        log_level="INFO",
        message="SELECT * FROM users WHERE id = $1",
        metadata={"query": "SELECT", "execution_time_ms": str(random.randint(5, 25))}
    )
    await asyncio.sleep(0.05)
    
    yield LogEvent(
        trace_id=trace_id,
        service_name="Frontend-Gateway",
        log_level="INFO",
        message="Successfully processed request, returning 200 OK",
        metadata={"status_code": "200"}
    )

async def generate_failure_cascade(trace_id: str) -> AsyncGenerator[LogEvent, None]:
    """
    Simulates a multi-layered cascading microservice failure.
    Example: DB pool depletion -> Auth timeout -> Gateway 500.
    """
    yield LogEvent(
        trace_id=trace_id,
        service_name="Frontend-Gateway",
        log_level="INFO",
        message="Received incoming HTTP POST request for /api/v1/checkout",
        metadata={"path": "/api/v1/checkout", "method": "POST"}
    )
    await asyncio.sleep(0.05)
    
    yield LogEvent(
        trace_id=trace_id,
        service_name="Payment-API",
        log_level="INFO",
        message="Initiating payment transaction sequence.",
        metadata={"action": "transaction_init"}
    )
    await asyncio.sleep(0.05)

    # The Root Cause: DB Connection Pool Exhaustion
    yield LogEvent(
        trace_id=trace_id,
        service_name="User-PostgreSQL",
        log_level="CRITICAL",
        message="FATAL: remaining connection slots are reserved for non-replication superuser connections. Pool exhausted.",
        metadata={"error_code": "53300", "state": "connection_limit_exceeded"}
    )
    await asyncio.sleep(0.3)
    
    # The Cascade 1: Auth Service Timeout
    yield LogEvent(
        trace_id=trace_id,
        service_name="Auth-Service",
        log_level="ERROR",
        message="TimeoutException: Failed to acquire sql connection, pool exhausted after 3000ms.",
        metadata={"retry_count": "3", "target": "User-PostgreSQL"}
    )
    await asyncio.sleep(0.1)

    # The Cascade 2: Payment API Failure
    yield LogEvent(
        trace_id=trace_id,
        service_name="Payment-API",
        log_level="ERROR",
        message="Transaction failed due to upstream authentication timeout.",
        metadata={"status": "FAILED", "dependency": "Auth-Service"}
    )
    await asyncio.sleep(0.05)
    
    # The Surface Impact: Gateway 500
    yield LogEvent(
        trace_id=trace_id,
        service_name="Frontend-Gateway",
        log_level="CRITICAL",
        message="500 Internal Server Error returned to client.",
        metadata={"status_code": "500", "client_ip": f"192.168.1.{random.randint(10,250)}"}
    )

async def simulate_traffic(num_traces: int = 10, failure_rate: float = 0.2):
    """
    Main simulator loop. Generates structured JSON traces for a given number of requests.
    Yields events asynchronously and publishes directly to RabbitMQ.
    """
    print(f"Starting Symantix Log Simulator (Environment: {settings.environment})")
    
    connection = await aio_pika.connect_robust("amqp://guest:guest@localhost/")
    async with connection:
        channel = await connection.channel()
        exchange = channel.default_exchange
        
        for i in range(num_traces):
            trace_id = str(uuid.uuid4())
            is_failure = random.random() < failure_rate
            
            if is_failure:
                generator = generate_failure_cascade(trace_id)
            else:
                generator = generate_healthy_logs(trace_id)
                
            async for log_event in generator:
                # Emit structured JSON to RabbitMQ
                message_body = log_event.model_dump_json().encode()
                await exchange.publish(
                    aio_pika.Message(
                        body=message_body,
                        content_type="application/json"
                    ),
                    routing_key="symantix_logs"
                )
                print(f"Published event for trace {trace_id} from {log_event.service_name}")
                
            # Add slight jitter between traces
            await asyncio.sleep(random.uniform(0.1, 0.5))

if __name__ == "__main__":
    try:
        asyncio.run(simulate_traffic(num_traces=5, failure_rate=0.4))
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
