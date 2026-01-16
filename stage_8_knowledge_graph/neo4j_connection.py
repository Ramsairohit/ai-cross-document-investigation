"""
Stage 8: Knowledge Graph - Neo4j Connection

Singleton pattern for Neo4j driver with connection management.

Uses Neo4j Community Edition only.
No APOC procedures.
No graph algorithms.
Parameterized Cypher queries only.
"""

import os
from contextlib import contextmanager
from threading import Lock
from typing import Any, Generator, Optional

from neo4j import Driver, GraphDatabase, Session


class Neo4jConnection:
    """
    Singleton Neo4j connection manager.

    Thread-safe singleton pattern ensures one driver instance.
    All queries use parameterized Cypher for security.
    """

    _instance: Optional["Neo4jConnection"] = None
    _lock: Lock = Lock()
    _driver: Optional[Driver] = None

    def __new__(cls) -> "Neo4jConnection":
        """
        Singleton pattern implementation.

        Returns:
            The singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize connection (no-op if already initialized)."""
        pass

    def connect(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        """
        Connect to Neo4j database.

        Args:
            uri: Neo4j connection URI (default: NEO4J_URI env var or bolt://localhost:7687)
            user: Neo4j username (default: NEO4J_USER env var or neo4j)
            password: Neo4j password (default: NEO4J_PASSWORD env var, required)
            database: Database name (default: NEO4J_DATABASE env var or neo4j)
        """
        if self._driver is not None:
            return  # Already connected

        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD")
        self._database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        if not self._password:
            raise ValueError("Neo4j password is required. Set NEO4J_PASSWORD environment variable.")

        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )

        # Verify connectivity
        self._driver.verify_connectivity()

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def get_driver(self) -> Driver:
        """
        Get the Neo4j driver instance.

        Returns:
            The Neo4j driver.

        Raises:
            RuntimeError: If not connected.
        """
        if self._driver is None:
            raise RuntimeError("Neo4j not connected. Call connect() first.")
        return self._driver

    @contextmanager
    def session(self, database: Optional[str] = None) -> Generator[Session, None, None]:
        """
        Context manager for Neo4j sessions.

        Args:
            database: Optional database name override.

        Yields:
            Neo4j session.
        """
        driver = self.get_driver()
        db = database or getattr(self, "_database", "neo4j")
        session = driver.session(database=db)
        try:
            yield session
        finally:
            session.close()

    def execute_query(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a parameterized Cypher query.

        Args:
            query: Cypher query string with $parameter placeholders.
            parameters: Query parameters.
            database: Optional database name override.

        Returns:
            List of result records as dictionaries.
        """
        with self.session(database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def execute_write(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a write transaction with parameterized Cypher.

        Args:
            query: Cypher query string with $parameter placeholders.
            parameters: Query parameters.
            database: Optional database name override.

        Returns:
            List of result records as dictionaries.
        """

        def _write_tx(tx: Any) -> list[dict[str, Any]]:
            result = tx.run(query, parameters or {})
            return [record.data() for record in result]

        with self.session(database) as session:
            return session.execute_write(_write_tx)

    def is_connected(self) -> bool:
        """
        Check if connected to Neo4j.

        Returns:
            True if connected, False otherwise.
        """
        return self._driver is not None


def get_connection() -> Neo4jConnection:
    """
    Get the Neo4j connection singleton.

    Returns:
        The Neo4jConnection instance.
    """
    return Neo4jConnection()


def connect_neo4j(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
) -> Neo4jConnection:
    """
    Connect to Neo4j and return the connection.

    Convenience function for initializing the connection.

    Args:
        uri: Neo4j connection URI.
        user: Neo4j username.
        password: Neo4j password.
        database: Database name.

    Returns:
        The connected Neo4jConnection instance.
    """
    conn = get_connection()
    conn.connect(uri=uri, user=user, password=password, database=database)
    return conn


def close_neo4j() -> None:
    """Close the Neo4j connection."""
    conn = get_connection()
    conn.close()
