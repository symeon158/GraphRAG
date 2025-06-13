import openai
import time
from neo4j_connector import neo4j_db
import os
from dotenv import load_dotenv

load_dotenv()

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

BATCH_SIZE = 100  # Προσαρμόστε ανάλογα με την απόδοση
def get_embeddings(texts):
    try:
        response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
        return [item["embedding"] for item in response["data"]]
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return [None] * len(texts)


def update_embeddings_in_bulk():
    """Ενημερώνει τους κόμβους που δεν έχουν embeddings σε παρτίδες."""
    cypher_query = "MATCH (n) WHERE n.embedding IS NULL RETURN n.name AS name, ID(n) AS id"
    nodes = neo4j_db.query(cypher_query)

    if not nodes:
        print("✅ Όλοι οι κόμβοι έχουν ήδη embeddings.")
        return

    # Διαχωρίζουμε τους κόμβους σε παρτίδες
    node_batches = [nodes[i:i+BATCH_SIZE] for i in range(0, len(nodes), BATCH_SIZE)]

    for batch in node_batches:
        names = [node["name"] for node in batch]
        ids = [node["id"] for node in batch]

        print(f"🔄 Επεξεργασία παρτίδας {len(names)} κόμβων...")
        embeddings = get_embeddings(names)

        # Ετοιμάζουμε τη λίστα των updates
        updates = []
        for i, embedding in enumerate(embeddings):
            if embedding:
                updates.append({"id": ids[i], "embedding": embedding})

        if updates:
            update_query = """
            UNWIND $updates as upd
            MATCH (n) WHERE ID(n) = upd.id
            SET n.embedding = upd.embedding, n:Node
            """
            neo4j_db.query(update_query, {"updates": updates})
            print(f"✅ Ενημερώθηκαν {len(updates)} embeddings.")

        time.sleep(1)  # Μικρή παύση για να αποφευχθούν τα rate limits

    print("🎯 Όλα τα missing embeddings ενημερώθηκαν στη Neo4j!")

if __name__ == '__main__':
    update_embeddings_in_bulk()
