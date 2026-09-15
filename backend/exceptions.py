class SymantixBaseException(Exception):
    """Base exception for all custom Symantix errors."""
    pass

class DatabaseConnectionError(SymantixBaseException):
    """Raised when the connection to a database (Neo4j/ChromaDB) fails."""
    pass

class InfrastructureDegradationError(SymantixBaseException):
    """Raised when an infrastructure component fails, triggering a fallback mechanism."""
    pass

class SimulationError(SymantixBaseException):
    """Raised when the logger simulator encounters an inconsistent state."""
    pass
