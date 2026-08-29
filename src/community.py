import networkx as nx
import igraph as ig
import leidenalg
from src.neo4j_client import get_driver


from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(
    model=os.getenv("LLM_MODEL"),
    model_provider="openai",
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS")),
)

def load_graph_from_neo4j():
    """Step 1: pull nodes + relationships out of Neo4j into a NetworkX graph."""
    driver = get_driver()
    G = nx.Graph()  # undirected community detection doesn't care about direction

    with driver.session() as session:
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN elementId(a) AS source, elementId(b) AS target
        """)
        for record in result:
            G.add_edge(record["source"], record["target"])

    print(f"Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges into NetworkX")
    return G


def run_leiden(G: nx.Graph):
    """Step 2: convert to igraph, run Leiden, return {node_id: community_id}."""
    node_list = list(G.nodes())
    node_index = {node: i for i, node in enumerate(node_list)}  # NetworkX id -> igraph index

    edges = [(node_index[u], node_index[v]) for u, v in G.edges()]
    ig_graph = ig.Graph(edges=edges, n=len(node_list))

    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

    # map back from igraph index -> original Neo4j node id -> community number
    community_map = {}
    for community_id, members in enumerate(partition):
        for idx in members:
            neo4j_id = node_list[idx]
            community_map[neo4j_id] = community_id

    print(f"Found {len(partition)} communities")
    return community_map


def write_communities_to_neo4j(community_map: dict):
    """Step 3: write community_id back onto each node as a property."""
    driver = get_driver()
    with driver.session() as session:
        for neo4j_id, community_id in community_map.items():
            session.run("""
                MATCH (n) WHERE elementId(n) = $id
                SET n.community_id = $community_id
            """, id=neo4j_id, community_id=community_id)
    print("Community IDs written back to Neo4j")



def get_community_ids():
    """Find every distinct community_id that exists."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (n) WHERE n.community_id IS NOT NULL
            RETURN DISTINCT n.community_id AS community_id
        """)
        ids = [record["community_id"] for record in result]
    return ids


def get_facts_for_community(community_id: int):
    """Pull all nodes + relationships belonging to one community."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (a)-[r]-(b)
            WHERE a.community_id = $cid AND b.community_id = $cid
            RETURN DISTINCT a.id AS source, type(r) AS relationship, b.id AS target
        """, cid=community_id)
        facts = [f"{row['source']} --{row['relationship']}--> {row['target']}" for row in result]
    return facts


def summarize_community(community_id: int):
    """Ask the LLM to write a short summary of one community's facts."""
    facts = get_facts_for_community(community_id)
    if not facts:
        return None

    facts_text = "\n".join(facts)
    prompt = f"""Here are facts from one cluster of a knowledge graph:
{facts_text}

Write a short (2-4 sentence) summary describing what this cluster is about overall."""

    response = llm.invoke(prompt)
    return response.content


def write_community_summary(community_id: int, summary: str):
    """Store the summary as its own node, linked conceptually via community_id."""
    driver = get_driver()
    with driver.session() as session:
        session.run("""
            MERGE (c:Community {community_id: $cid})
            SET c.summary = $summary
        """, cid=community_id, summary=summary)


def build_all_community_summaries():
    """Run the full summarization pass: one summary per community."""
    ids = get_community_ids()
    print(f"Summarizing {len(ids)} communities...")
    for cid in ids:
        summary = summarize_community(cid)
        if summary:
            write_community_summary(cid, summary)
            print(f"Community {cid}: {summary[:80]}...")
        else:
            print(f"Community {cid}: no facts found, skipped")