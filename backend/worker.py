import asyncio
import json
import logging
import aio_pika
from pydantic import ValidationError

from .models import LogEvent
from .graph_extractor import LogEntityExtractor
from .graph_store import AsyncGraphManager
from .vector_store import AsyncChromaManager

import re
logger = logging.getLogger(__name__)

def sanitize_log(message: str) -> str:
    """Redact potential secrets (JWTs, Bearer tokens, passwords) before processing."""
    # Redact Bearer tokens
    message = re.sub(r"Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*", "Bearer [REDACTED]", message)
    # Redact potential passwords
    message = re.sub(r"(?i)(password|secret|key|token)[\s:=]+[\"']?[^\s\"']+[\"']?", r"\1 [REDACTED]", message)
    return message

async def process_message(message: aio_pika.abc.AbstractIncomingMessage, 
                          graph_manager: AsyncGraphManager, 
                          chroma_manager: AsyncChromaManager,
                          extractor: LogEntityExtractor):
    async with message.process():
        try:
            # Parse payload strictly using Pydantic
            payload = json.loads(message.body.decode())
            log_event = LogEvent(**payload)
            
            # Sanitize log message to prevent secret leakage into Vector/Graph stores
            safe_message = sanitize_log(log_event.message)
            
            # Extract Graph Nodes via spaCy
            entities = extractor.extract(safe_message, log_event.metadata)
            
            # Generate deterministic Vector ID
            vector_id = f"{log_event.trace_id}_{log_event.timestamp.timestamp()}"
            
            # 1. Embed and Upsert to ChromaDB concurrently
            await chroma_manager.ingest_logs(
                log_ids=[vector_id],
                log_messages=[safe_message],
                metadata_list=[{
                    "trace_id": log_event.trace_id,
                    "service_name": log_event.service_name,
                    "log_level": log_event.log_level
                }]
            )
            
            # 2. Ingest to Neo4j GraphDB concurrently
            await graph_manager.ingest_log_entities(
                trace_id=log_event.trace_id,
                service_name=log_event.service_name,
                timestamp=str(log_event.timestamp),
                extracted_entities=entities,
                vector_id=vector_id
            )
            
            logger.info(f"Processed trace {log_event.trace_id} from {log_event.service_name}")
            
        except ValidationError as e:
            logger.error(f"Payload validation failed (Dropping Message): {e}")
        except Exception as e:
            logger.error(f"Worker encountered processing error: {e}")

async def main():
    logger.info("Initializing RabbitMQ connection...")
    # In production, use env config for AMQP URI
    from .config import settings
    connection = await aio_pika.connect_robust(settings.amqp_uri)
    channel = await connection.channel()
    
    # Set QoS prefetch to prevent memory overload during massive log spikes
    await channel.set_qos(prefetch_count=50)
    
    queue = await channel.declare_queue("symantix_logs", durable=True)
    
    graph_manager = AsyncGraphManager()
    chroma_manager = AsyncChromaManager()
    
    logger.info("Initializing Database Schemas...")
    await graph_manager.initialize_schema()
    await chroma_manager.initialize_collection()
    
    extractor = LogEntityExtractor()
    
    logger.info("Worker started. Listening for messages on 'symantix_logs' queue...")
    
    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            # Dispatch processing to the event loop asynchronously
            asyncio.create_task(
                process_message(message, graph_manager, chroma_manager, extractor)
            )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker gracefully shutting down.")
