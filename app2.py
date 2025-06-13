import streamlit as st
import os
from dotenv import load_dotenv
from neo4j_connector import neo4j_db  # Using the existing Neo4j connection
from pyvis.network import Network
import streamlit.components.v1 as components
from query_neo4j import get_graph_data, hybrid_search
from simple_rag import simple_rag_query, init_chroma_collection
from hybrid_rag import hybrid_simple_graph_search
from llm_response import generate_response
from Levenshtein import ratio
from hybrid_rag import enhanced_hybrid_search


# Load environment variables
load_dotenv()

def safe_rerun():
    """Reruns the Streamlit app if possible."""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def show_impressive_title():
    title_html = """
    <div style="
        background: linear-gradient(to right, #2c5364, #203a43, #0f2027);
        padding: 2em;
        text-align: center;
        border-radius: 6px;
        width: 100%;
        max-width: 1200px;
        margin: 2em auto;
        font-family: 'Montserrat', sans-serif;
      ">
      <h1 style="
        color: #f2f2f2;
        font-size: 2.5em;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      ">
        Constructing a Chatbot for Public Services
      </h1>
      <p style="
        color: #fafafa;
        font-size: 1.1em;
        margin-top: 0.8em;
        font-weight: 300;
      ">
        Using <strong>LLMs</strong> and <strong>Knowledge Graphs</strong>: 
        <em>Catalogue of Services</em>
      </p>
    </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)

############################################
# Neo4j Query Helpers
############################################

def get_topic_list():
    query = """
        MATCH (t:TOPIC)
        WHERE t.name IS NOT NULL
        RETURN DISTINCT t.name AS topicName
        ORDER BY t.name
    """
    return [r["topicName"] for r in neo4j_db.query(query)]

def get_node_list_by_topic(selected_topic: str):
    if selected_topic:
        query = """
            MATCH (n:PROCESS)-[:HAS_TOPIC]->(t:TOPIC {name: $selected_topic})
            WHERE n.name IS NOT NULL
            RETURN DISTINCT n.name AS nodeName
            ORDER BY n.name
        """
        result = neo4j_db.query(query, {"selected_topic": selected_topic})
    else:
        query = """
            MATCH (n:PROCESS)
            WHERE n.name IS NOT NULL
            RETURN DISTINCT n.name AS nodeName
            ORDER BY n.name
        """
        result = neo4j_db.query(query)
    return [r["nodeName"] for r in result]

############################################
# Graph Visualization
############################################

class KnowledgeGraphRAG:
    def __init__(self):
        self.driver = neo4j_db.driver

    def create_3d_graph_for_node(self, selected_node: str):
        query = """
            MATCH (n)-[r]-(m)
            WHERE n.name = $selected_node AND m.name IS NOT NULL
            RETURN n.name AS source, type(r) AS relationship, m.name AS target
        """
        records = neo4j_db.query(query, {"selected_node": selected_node})
        if not records:
            return None

        net = Network(height="600px", width="100%", directed=True, notebook=False)
        net.barnes_hut()
        net.add_node(selected_node, label=selected_node, color="#FF5733", size=25)
        for rec in records:
            net.add_node(rec["source"], label=rec["source"])
            net.add_node(rec["target"], label=rec["target"])
            net.add_edge(rec["source"], rec["target"], title=rec["relationship"])
        net.repulsion(node_distance=200, central_gravity=0.3,
                      spring_length=100, spring_strength=0.05)
        net.set_options("""
        var options = {
          "interaction": {
            "navigationButtons": true,
            "zoomView": true
          }
        }
        """)
        return net.generate_html(notebook=False)

############################################
# RAG + Suggestion Helpers
############################################

collection = init_chroma_collection()
ALL_NODES = [r["nodeName"] for r in neo4j_db.query("MATCH (n:PROCESS) RETURN DISTINCT n.name AS nodeName")]

def suggest_nodes(query, candidates, limit=5):
    scores = [(cand, ratio(query, cand)) for cand in candidates]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [cand for cand, _ in scores[:limit]]

def paste_suggestion_to_buffer(node_name: str):
    """Callback: store into the 'buffer' that our chat_input logic will pick up."""
    st.session_state["chat_input"] = node_name

############################################
# Main
############################################

def main():
    st.set_page_config(page_title="Chatbot with Knowledge Graph Visualization", layout="wide")
    show_impressive_title()

    # --- Sidebar ---
    with st.sidebar:
        mode = st.radio("Choose Mode:", [
            "Graph RAG (Text-Only)",
            "Hybrid Graph (Text+Vector)",
            "Simple RAG (ChromaDB)",
            "Hybrid Simple + Graph"
        ], index=0)

        st.markdown("---")
        st.markdown("### Select a Topic")
        topics = get_topic_list()
        if topics:
            sel_topic = st.selectbox("Topic:", topics)
            st.markdown("### Select a Node")
            nodes = get_node_list_by_topic(sel_topic)
            if nodes:
                sel_node = st.selectbox("Node:", nodes)
            else:
                sel_node = None
                st.info("No nodes for this topic.")
        else:
            sel_topic = sel_node = None
            st.info("No topics in DB.")

        if sel_node and st.button("Paste Selected Node into Query"):
            st.session_state["chat_input"] = sel_node
            safe_rerun()

        if st.button("Reset Conversation"):
            st.session_state["messages"] = []
            safe_rerun()

    # --- Graph Visualization ---
    st.markdown("### 🌐 Knowledge Graph Visualization")
    if sel_node:
        html = KnowledgeGraphRAG().create_3d_graph_for_node(sel_node)
        if html:
            components.html(html, height=600, width=800)
        else:
            st.info("No relationships found.")
    else:
        st.info("Select a topic & node first.")

    # --- Chat Interface ---
    st.markdown('<div class="advanced-chatbot-heading">🤖 Intelligent Conversational Assistant</div>', unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat_input" not in st.session_state:
        st.session_state["chat_input"] = ""

    # Show history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # The actual input widget (read‐only for session_state)
    user_input = st.chat_input("Type your question in Greek:", key="chat_input_widget")

    # If widget is empty but buffer has a suggestion, use it
    if not user_input and st.session_state["chat_input"]:
        user_input = st.session_state["chat_input"]
        st.session_state["chat_input"] = ""

    # Process the input
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Processing your query..."):
            if mode == "Graph RAG (Text-Only)":
                data = get_graph_data(user_input)
            elif mode == "Hybrid Graph (Text+Vector)":
                data = hybrid_search(user_input, top_k=5)
            elif mode == "Simple RAG (ChromaDB)":
                docs = simple_rag_query(user_input, collection)
                data = [{"node_1": d, "relationship": "—", "node_2": ""} for d in docs]
            elif mode == "Hybrid Simple + Graph":
                data = enhanced_hybrid_search(user_input, top_k=5)
            else:
                data = []

            if not data:
                st.info("⚠️ Δεν εντοπίστηκε σχετική πληροφορία στη βάση γνώσης. Παρακαλώ διατυπώστε ερώτηση σχετική με δημόσιες υπηρεσίες.")
                answer = "⚠️ Η ερώτησή σας δεν αντιστοιχεί σε κάποια διαθέσιμη διοικητική υπηρεσία ή διαδικασία."
                suggestions = suggest_nodes(user_input, ALL_NODES, limit=5)
                for s in suggestions:
                    st.button(label=s, key=f"suggest_{s}", on_click=paste_suggestion_to_buffer, args=(s,))
                answer = "Παρακαλώ επιλέξτε μία από τις παραπάνω προτάσεις για καλύτερη αναζήτηση."
            else:
                answer = generate_response(user_input, data)


        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)


if __name__ == "__main__":
    main()
