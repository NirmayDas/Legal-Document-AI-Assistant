from neo4j import GraphDatabase

driver = GraphDatabase.driver("neo4j+s://<your-neo4j-host>", auth=("neo4j", "<password>"))

def upload_node_with_embedding(tx, name, embedding):
    tx.run("""
        CREATE (n:Item {name: $name, embedding: $embedding})
    """, name=name, embedding=embedding)

def upload_embedding(name, embedding):
    with driver.session() as session:
        session.write_transaction(upload_node_with_embedding, "My Item", embedding)
