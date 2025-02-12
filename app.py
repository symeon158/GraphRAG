import streamlit as st
from query_neo4j import get_graph_data, hybrid_search
from llm_response import generate_response

def show_impressive_title():
    # HTML + inline CSS for a gradient title section
    title_html = """
    <div style="
        background: linear-gradient(to right, #2c5364, #203a43, #0f2027);
        padding: 2em;
        text-align: center;
        border-radius: 6px;
        width: 100%;  /* Occupies entire viewport width */
        max-width: 1200px; /* Prevents lines from getting too long on huge screens */
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
        <em>The Greek National Catalogue of Services (MITOS)</em>
      </p>
    </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)

def main():
    #st.title("Compare Normal (Text-Only) Mode vs. Hybrid (Text+Vector) Search")
    #st.set_page_config(page_title="Chatbot with LLMs & Knowledge Graphs", layout="centered")

    # Show the styled title
    show_impressive_title()

    # 1. Initialize the messages in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 2. Choose your retrieval mode
    mode = st.sidebar.radio("Choose Mode:", ["Graph RAG (Text-Only)", "Hybrid (Text+Vector)"], index=0)

    # 3. Display the existing conversation
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 4. Use st.chat_input instead of st.text_input
    user_input = st.chat_input("Ask a question in Greek:")
    # st.chat_input clears itself automatically on send, no need to do st.session_state["user_input"] = ""

    # 5. When there's new user input, handle it
    if user_input:
        # A) Display user's message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # B) Retrieve data from Neo4j, generate final response
        if mode == "Graph RAG (Text-Only)":
            data = get_graph_data(user_input)
            if data:
                answer = generate_response(user_input, data, mode="text_only")
            else:
                answer = "No relevant information found in text-only mode."
        else:  # "Hybrid (Text+Vector)"
            data = hybrid_search(user_input, top_k=5)
            if data:
                answer = generate_response(user_input, data, mode="hybrid")
            else:
                answer = "No relevant information found in hybrid mode."

        # C) Display assistant's response
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

if __name__ == "__main__":
    main()
