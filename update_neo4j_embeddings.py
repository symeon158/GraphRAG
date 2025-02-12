import openai
import time
import numpy as np
from neo4j_connector import neo4j_db
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Batch size for processing embeddings
BATCH_SIZE = 100  # Adjust based on performance

def get_embeddings(texts):
    """Generate embeddings in batches to avoid rate limits."""
    try:
        response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
        return [r.embedding for r in response.data]
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return [None] * len(texts)  # Return empty embeddings if API fails

def update_embeddings_in_bulk():
    """Update missing embeddings in Neo4j in bulk."""
    cypher_query = "MATCH (n) WHERE n.embedding IS NULL RETURN n.name AS name, ID(n) AS id"
    nodes = neo4j_db.query(cypher_query)

    if not nodes:
        print("✅ All nodes already have embeddings.")
        return

    node_batches = [nodes[i:i+BATCH_SIZE] for i in range(0, len(nodes), BATCH_SIZE)]

    for batch in node_batches:
        names = [node["name"] for node in batch]
        ids = [node["id"] for node in batch]

        print(f"🔄 Processing batch of {len(names)} nodes...")
        embeddings = get_embeddings(names)

        # Prepare bulk Cypher update query
        update_statements = []
        for i, embedding in enumerate(embeddings):
            if embedding:
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                update_statements.append(f"WHEN ID(n) = {ids[i]} THEN SET n.embedding = {embedding_str}")

        if update_statements:
            update_query = "MATCH (n) WHERE " + " ".join(update_statements)
            neo4j_db.query(update_query)
            print(f"✅ Successfully updated {len(update_statements)} embeddings.")

        time.sleep(1)  # Prevent rate limits from OpenAI

    print("🎯 All missing embeddings updated in Neo4j!")

# Run bulk update
update_embeddings_in_bulk()
