import spacy
from spacy.matcher import Matcher
from typing import Dict, List, Optional
from pydantic import BaseModel

class ExtractedEntities(BaseModel):
    services: List[str]
    resources: List[str]
    error_states: List[str]
    dependencies: List[str]

class LogEntityExtractor:
    """
    Rule-and-token-based entity extraction engine for SRE logs.
    Utilizes spaCy to map natural language and structured logs to topological components.
    """
    def __init__(self):
        # Load the base english model (requires: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback for demonstration if model is not downloaded; in prod, this would fail fast
            import spacy.blank
            self.nlp = spacy.blank("en")
            
        self.matcher = Matcher(self.nlp.vocab)
        self._initialize_rules()

    def _initialize_rules(self):
        """
        Defines explicit token matching rules for known SRE infrastructure parameters.
        """
        # Service Rules
        service_pattern = [{"LOWER": {"IN": ["frontend-gateway", "auth-service", "payment-api", "user-postgresql"]}}]
        self.matcher.add("SERVICE", [service_pattern])
        
        # Error Types
        error_pattern = [{"TEXT": {"REGEX": "^[A-Z][a-zA-Z]*Exception$"}}]
        error_pattern2 = [{"LOWER": {"IN": ["timeout", "fatal", "exhausted", "failed"]}}]
        self.matcher.add("ERROR_STATE", [error_pattern, error_pattern2])
        
        # Resources (e.g. Database Pools)
        resource_pattern = [{"LOWER": "pool"}]
        self.matcher.add("RESOURCE", [resource_pattern])

    def extract(self, message: str, metadata: Optional[Dict[str, str]] = None) -> ExtractedEntities:
        """
        Processes a log message to extract nodes for the GraphRAG pipeline.
        """
        doc = self.nlp(message)
        matches = self.matcher(doc)
        
        services = set()
        resources = set()
        errors = set()
        dependencies = set()
        
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            if label == "SERVICE":
                services.add(span.text)
            elif label == "ERROR_STATE":
                errors.add(span.text)
            elif label == "RESOURCE":
                resources.add(span.text)
                
        # Also extract from metadata
        if metadata:
            if "dependency" in metadata:
                dependencies.add(metadata["dependency"])
            if "target" in metadata:
                dependencies.add(metadata["target"])
                
        return ExtractedEntities(
            services=list(services),
            resources=list(resources),
            error_states=list(errors),
            dependencies=list(dependencies)
        )
