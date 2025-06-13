from query_neo4j import hybrid_search, full_text_search, get_graph_data
from simple_rag import simple_rag_query, init_chroma_collection

collection = init_chroma_collection()

def hybrid_simple_graph_search(user_query, top_k=5):
    # Simple RAG (ChromaDB)
    simple_results = simple_rag_query(user_query, collection, top_k)
    simple_data = [{"node_1": doc, "relationship": "SIMPLE_RAG", "node_2": ""} for doc in simple_results]

    # Graph RAG (Neo4j)
    graph_results = hybrid_search(user_query, top_k=top_k)

    combined_context = simple_data + graph_results

    # Deduplicate clearly based on nodes
    seen = set()
    unique_results = []
    for item in combined_context:
        key = (item["node_1"], item["relationship"], item["node_2"])
        if key not in seen:
            unique_results.append(item)
            seen.add(key)
        if len(unique_results) >= top_k:
            break

    return unique_results

def enhanced_hybrid_search(user_query, top_k=5):
    vector_graph_results = hybrid_search(user_query, top_k)
    simple_results = simple_rag_query(user_query, collection, top_k)
    fulltext_results = full_text_search(user_query, top_k)
    graph_results = get_graph_data(user_query)

    combined_results = vector_graph_results + fulltext_results + graph_results + \
                       [{"node_1": doc, "relationship": "SIMPLE_RAG", "node_2": ""} for doc in simple_results]

    seen = set()
    unique_results = []
    for item in combined_results:
        key = (item["node_1"], item["relationship"], item["node_2"])
        if key not in seen:
            unique_results.append(item)
            seen.add(key)
        if len(unique_results) >= top_k:
            break

    return unique_results
