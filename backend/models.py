from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Literal, Optional

class LogEvent(BaseModel):
    """
    Strictly typed Pydantic model for generated system logs.
    Ensures all ingestion data structures are validated.
    """
    trace_id: str = Field(..., description="Unique identifier for the request trace.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of the log event.")
    service_name: str = Field(..., description="Name of the microservice emitting the log.")
    log_level: Literal["INFO", "WARN", "ERROR", "CRITICAL"] = Field(..., description="Severity level of the log.")
    message: str = Field(..., description="The raw log message.")
    metadata: Optional[dict[str, str]] = Field(default_factory=dict, description="Additional structured context (e.g. status_code, query_string).")
