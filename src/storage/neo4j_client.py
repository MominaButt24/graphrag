from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph

from src.config.settings import settings

_driver = None  # module-level singleton, created once

def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
    return _driver

def close_driver():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None

def get_graph():
    return Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )


def delete_graph_document(document_id: str):
    """Delete all graph nodes and relations tagged with the same document_id property."""
    driver = get_driver()
    with driver.session() as session:
        session.run(
            """
            MATCH ()-[r]->()
            WHERE r.document_id = $document_id
            DELETE r
            """,
            {"document_id": document_id},
        )
        session.run(
            """
            MATCH (n)
            WHERE n.document_id = $document_id
            DETACH DELETE n
            """,
            {"document_id": document_id},
        )