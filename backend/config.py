import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class EnvironmentConfig(BaseSettings):
    """
    Singleton Configuration for Symantix System.
    Strictly follows 12-Factor App design. All variables are loaded from environment.
    """
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Neo4j Settings
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    
    # ChromaDB Settings
    chroma_host: str
    chroma_port: int
    
    # App Settings
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    api_key: str = "default_unsafe_key_override_in_production"
    amqp_uri: str = "amqp://guest:guest@localhost/"

# Singleton instance
settings = EnvironmentConfig()
