from src.neo4j_client import get_driver

driver = get_driver()
with driver.session() as session:
    result = session.run("RETURN 'connection ok' AS status")
    print(result.single()["status"])
driver.close()