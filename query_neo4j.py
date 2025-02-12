# query_neo4j.py
import os
import openai
from dotenv import load_dotenv
from neo4j_connector import neo4j_db

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_graph_data(user_query):
    """
    Normal text-only search:
    Returns nodes and relationships where node.name contains user query.
    """
    cypher_query = f"""
    MATCH (n)-[r]->(m) 
    WHERE n.name CONTAINS '{user_query}' OR m.name CONTAINS '{user_query}'
    RETURN n, r, m LIMIT 20
    """
    results = neo4j_db.query(cypher_query)
    
    extracted_data = [
        {
            "node_1": record["n"]["name"],
            "relationship": record["r"].type,
            "node_2": record["m"]["name"]
        } 
        for record in results
    ]
    return extracted_data

# Initialize the OpenAI client with your API key
client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding

def hybrid_search(user_query, top_k=5):
    """
    Hybrid approach that searches:
    1) Vector similarity
    2) Text/fuzzy on node.name
    3) Keyword-based matching
    """
    user_embedding = get_embedding(user_query)  # Generate embedding for user query

    cypher_query = """
    CALL {
  // 1) Vector Similarity
  CALL db.index.vector.queryNodes('vector_index', $top_k, $user_embedding)
  YIELD node, score
  RETURN node, score
  
  UNION
  
  // 2) Text/Fuzzy with morphological normalization
  MATCH (node)
  WITH node,
       apoc.text.regreplace(node.name, "(ος|ης|ων|ση|σης|ώσεις)$", "") AS normName,
       apoc.text.regreplace($user_query, "(ος|ης|ων|ση|σης|ώσεις)$", "") AS normQuery
  WHERE apoc.text.levenshteinDistance(normName, normQuery) < 4
        OR normName CONTAINS normQuery
  RETURN node, 1.0 AS score

  UNION
  
  // 3) Keyword-based matches
  MATCH (node)-[:HAS_KEYWORD]->(k:Keyword)
  WHERE apoc.text.levenshteinDistance(k.name, $user_query) < 4
        OR k.name CONTAINS $user_query
  RETURN node, 1.0 AS score
}
RETURN DISTINCT node, score
ORDER BY score DESC
LIMIT $top_k

    """

    results = neo4j_db.query(cypher_query, {
        "top_k": top_k,
        "user_embedding": user_embedding,
        "user_query": user_query
    })

    # Format results for your LLM usage
    extracted_data = []
    for record in results:
        extracted_data.append({
            "node_1": record.get("name", ""),
            "relationship": "similar",
            "node_2": record.get("description", ""),
            "score": record["score"]
        })
    return extracted_data

