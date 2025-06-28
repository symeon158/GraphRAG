# Chatbot for Public Services Using LLMs and Knowledge Graphs

Constructing a Chatbot for Public Services is a project that leverages Large Language Models (LLMs) and Neo4j Knowledge Graphs to provide enhanced, user-friendly access to information from the Greek National Catalogue of Services (MITOS). The application is built using Streamlit and offers both a conversational chat interface and interactive graph visualization.

## 🎥Demo Video

Watch the demo video [here](https://1drv.ms/v/c/A9927BE78AA24F21/ET2EPfpCY2pKmFE0TJ0U154B_g8wpTVcjTHeUmcY_Wudrw?e=zFav3f).


## Features

- **Conversational Chatbot Interface**  
  Multi-turn chat using Streamlit’s `st.chat_input` and `st.chat_message` components.

- **Hybrid Retrieval Modes**  
  Choose between:
  - **Graph RAG (Text-Only):** Uses traditional text matching.
  - **Hybrid (Text+Vector):** Combines semantic vector search with fuzzy and keyword matching for more robust retrieval.

- **Interactive Neo4j Graph Visualization**  
  Input a Cypher query via the sidebar to visualize a subgraph from Neo4j using PyVis with an enhanced, modern design.

- **Gradient Title & Polished UI**  
  The application features a modern gradient title section to present a professional look.

## Technologies

- **Streamlit** – Frontend and interactive UI.
- **Neo4j** – Graph database for storing public service data.
- **PyVis** – For interactive graph visualizations.
- **OpenAI** – To generate embeddings (using models like `text-embedding-ada-002`).
- **LangChain** – (Optional) To integrate LLM-based responses.
- **APOC** – Neo4j plugin for fuzzy matching and text processing.
- **Python-dotenv** – For managing environment variables.

## Project Structure

. ├── app.py # Main Streamlit application (chat and graph visualization) ├── query_neo4j.py # Functions to query Neo4j (Graph RAG and Hybrid search) ├── llm_response.py # Functions to generate responses using LLMs ├── neo4j_connector.py # Neo4j database connector code ├── requirements.txt # Project dependencies └── README.md # This file
