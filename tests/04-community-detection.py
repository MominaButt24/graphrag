from src.community import load_graph_from_neo4j, run_leiden, write_communities_to_neo4j

G = load_graph_from_neo4j()
community_map = run_leiden(G)
write_communities_to_neo4j(community_map)