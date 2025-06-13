from neo4j_connector import neo4j_db
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def full_text_search(user_query, top_k=5):
    cypher_query = """
    CALL db.index.fulltext.queryNodes('mitosFullTextIndex', $user_query) 
    YIELD node, score
    RETURN node, score
    ORDER BY score DESC
    LIMIT $top_k
    """
    records = neo4j_db.query(cypher_query, {
    "user_query": user_query,
    "top_k": top_k
})
    
    return [
        {
            "node_1": record["node"]["name"],
            "relationship": "MATCHED_BY_FULLTEXT",
            "node_2": f"Score: {record['score']:.2f}"
        }
        for record in records
    ]

def get_graph_data(user_query):
    cypher_query = """
    MATCH (startNode)
    WHERE startNode.name CONTAINS $user_query
    CALL apoc.path.subgraphAll(startNode, {
        maxLevel: 3,
        relationshipFilter: ">|<"
    })
    YIELD relationships
    UNWIND relationships AS r
    RETURN DISTINCT
        startNode.name AS node_1,
        type(r) AS relationship,
        endNode(r).name AS node_2
    LIMIT 200
    """
    records = neo4j_db.query(cypher_query, {"user_query": user_query})
    return [{"node_1": rec["node_1"], "relationship": rec["relationship"], "node_2": rec["node_2"]} for rec in records]

def get_embedding(text):
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

def hybrid_search(user_query: str, top_k: int = 5) -> list[dict]:
    user_embedding = get_embedding(user_query)
    cypher = """
    CALL {
      CALL db.index.vector.queryNodes('vector_index', $top_k, $user_embedding)
      YIELD node, score RETURN node, score
      UNION
      CALL db.index.fulltext.queryNodes('mitosFullTextIndex', $user_query)
      YIELD node, score RETURN node, score
    }
    WITH node, score ORDER BY score DESC LIMIT $top_k
    CALL apoc.path.subgraphAll(node, { maxLevel: 3, relationshipFilter: ">|<" })
    YIELD relationships
    UNWIND relationships AS r
    RETURN DISTINCT
        node.name AS node_1,
        type(r) AS relationship,
        endNode(r).name AS node_2
    LIMIT 100
    """
    records = neo4j_db.query(cypher, {"top_k": top_k, "user_embedding": user_embedding, "user_query": user_query})
    return [{"node_1": rec["node_1"], "relationship": rec["relationship"], "node_2": rec["node_2"]} for rec in records]
