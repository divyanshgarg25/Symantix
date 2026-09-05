import asyncio
import logging
from neo4j import AsyncGraphDatabase, exceptions as neo4j_exceptions
from typing import List, Dict, Any, Optional

from .config import settings
from .exceptions import DatabaseConnectionError, InfrastructureDegradationError
from .graph_extractor import ExtractedEntities

logger = logging.getLogger(__name__)

class AsyncGraphManager:
    """
    Asynchronous connection manager for Neo4j.
    Enforces the Graceful Degradation & Circuit Breaking rule.
    """
    def __init__(self):
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password
        self.driver = None

    async def connect(self):
        """Initializes the async driver."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            await self.driver.verify_connectivity()
        except neo4j_exceptions.ServiceUnavailable as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise DatabaseConnectionError(f"Failed to connect to Neo4j: {e}")

    async def close(self):
        """Closes the async driver."""
        if self.driver:
            await self.driver.close()

    async def initialize_schema(self):
        """
        Creates mandatory indexes on startup for trace_id and service_name.
        """
        if not self.driver:
            await self.connect()
            
        cypher_service_index = "CREATE INDEX service_name_idx IF NOT EXISTS FOR (s:Service) ON (s.name)"
        cypher_trace_index = "CREATE INDEX trace_id_idx IF NOT EXISTS FOR (e:Error) ON (e.trace_id)"
        
        try:
            async with self.driver.session() as session:
                await session.run(cypher_service_index)
                await session.run(cypher_trace_index)
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j schema: {e}")
            raise DatabaseConnectionError(f"Schema initialization failed: {e}")

    async def ingest_log_entities(self, trace_id: str, service_name: str, timestamp: str, 
                                 extracted_entities: ExtractedEntities, vector_id: str):
        """
        Asynchronously executes parameterized Cypher queries to build the topology graph.
        Stores the ChromaDB vector_id for cross-database referencing.
        """
        if not self.driver:
            try:
                await self.connect()
            except DatabaseConnectionError:
                # Graceful Degradation: Fallback mechanism triggered
                raise InfrastructureDegradationError("Neo4j is down. Falling back to vector-only retrieval.")

        cypher_query = """
        // Ensure Service node exists
        MERGE (s:Service {name: $service_name})
        
        // Register the Error Event
        MERGE (e:Error {trace_id: $trace_id})
        SET e.timestamp = $timestamp, e.vector_id = $vector_id
        
        // Link Service to Error
        MERGE (s)-[:EMITTED {timestamp: $timestamp}]->(e)
        
        // Register specific Error Types (from SpaCy)
        WITH s, e
        UNWIND $error_states AS err_type
        MERGE (et:ErrorType {name: err_type})
        MERGE (e)-[:IS_TYPE]->(et)
        
        // Register dependencies (from SpaCy)
        WITH s, e
        UNWIND $dependencies AS dep
        MERGE (ds:Service {name: dep})
        MERGE (s)-[:DEPENDS_ON]->(ds)
        """
        
        parameters = {
            "trace_id": trace_id,
            "service_name": service_name,
            "timestamp": timestamp,
            "vector_id": vector_id,
            "error_states": extracted_entities.error_states,
            "dependencies": extracted_entities.dependencies
        }
        
        try:
            async with self.driver.session() as session:
                await session.run(cypher_query, parameters)
        except Exception as e:
            logger.error(f"Neo4j ingestion failed for trace {trace_id}: {e}")
            raise InfrastructureDegradationError(f"Failed to write to GraphDB: {e}")

    async def traverse_topology(self, trace_ids: List[str], hops: int = 2) -> List[Dict[str, Any]]:
        """
        Executes a multi-hop traversal to retrieve structural dependencies for incident context.
        """
        if not self.driver:
            await self.connect()
            
        cypher_query = f"""
        MATCH (e:Error)
        WHERE e.trace_id IN $trace_ids
        MATCH (e)<-[:EMITTED]-(s:Service)
        MATCH path = (s)-[:DEPENDS_ON*1..{hops}]-(related:Service)
        RETURN s.name AS root_service, e.trace_id AS trace_id, 
               [x IN nodes(path) | x.name] AS dependency_chain
        """
        
        try:
            async with self.driver.session() as session:
                result = await session.run(cypher_query, trace_ids=trace_ids)
                records = await result.data()
                return records
        except Exception as e:
            logger.error(f"Neo4j traversal failed: {e}")
            raise InfrastructureDegradationError(f"Graph traversal failed: {e}")
