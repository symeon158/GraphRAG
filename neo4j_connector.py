from neo4j import GraphDatabase

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, cypher_query, parameters=None):
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [record for record in result]

# Replace with your Neo4j credentials
neo4j_db = Neo4jConnector(uri="bolt://localhost:7687", user="neo4j", password="Kaval@85*")
