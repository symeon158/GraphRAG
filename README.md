# 🤖 MITOS Ontology Graph-RAG Chatbot
### AI-Powered Public Services Assistant for the Greek National Catalogue of Services

An advanced **Ontology Graph-RAG** chatbot that makes Greek public services genuinely accessible to citizens — even when queries are informal, misspelled, or morphologically varied. Built on a **Neo4j knowledge graph** with intent-driven traversal, RRF score fusion, cross-encoder reranking, and streaming LLM responses.

---

## 🎥 Demo Video
[▶️ Watch the demo video](https://www.linkedin.com/feed/update/urn:li:activity:7454507172006264832/)

---

## 🏗️ Architecture

```
User Query (Greek)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  1. Query Understanding                               │
│     Greek normalization · fuzzy matching ·            │
│     intent classification                             │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│  2. Ontology-Aware Graph Traversal                    │
│     Intent → relationship type mapping:               │
│     "What documents?" → HAS_EVIDENCE                 │
│     "How to apply?"   → HAS_STEP → HAS_NEXT_STEP     │
│     "Where to go?"    → HAS_PROVISION_ORG            │
│     "Am I eligible?"  → HAS_CONDITION                │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│  3. Hybrid Retrieval + RRF Score Fusion               │
│     Vector search · Fulltext search · Graph signals  │
│     Reciprocal Rank Fusion (RRF) normalization        │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│  4. Cross-Encoder Reranking                           │
│     Precision-focused reranking of top candidates     │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│  5. LLM Response Generation (Streaming)               │
│     Multi-turn conversation memory                    │
│     Source citations linked to graph node IDs         │
└───────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🧠 Ontology-Aware Retrieval
The system goes far beyond keyword matching. It understands **query intent** and traverses the knowledge graph through the semantically correct relationship type — delivering precise, grounded answers rather than generic search results.

### 🔀 RRF Score Fusion
Combines three heterogeneous signals — vector embeddings, fulltext search, and graph traversal scores — using **Reciprocal Rank Fusion (RRF)**, which normalizes incompatible score ranges (e.g., vector: 0–1 vs fulltext: 0–∞) into a single reliable ranking.

### 🔁 Cross-Encoder Reranking
After retrieval, a **cross-encoder model** re-scores each candidate against the original query for precision-focused reranking — dramatically improving answer quality, especially for ambiguous Greek queries.

### 🇬🇷 Greek Morphology Handling
Greek is a highly inflected language. The system handles morphological variation through **query normalization, fuzzy matching (Levenshtein)**, and precomputed `HAS_SEM_SIMILAR_*` edges in the graph — so queries like *"πιστοποιητικό"* correctly match *"πιστοποιητικά"*.

### 📡 Streaming Responses
Responses stream token-by-token with real-time status updates: *analyzing → retrieving → reranking → expanding → writing* — giving users transparency into the pipeline.

### 🌐 Interactive D3.js Graph Visualization
An interactive **D3.js** graph renders the live knowledge graph directly in the app — letting users explore relationships between services, conditions, steps, and organizations behind every answer.

### 💬 Multi-Turn Conversation Memory
Full conversation history is passed to the LLM at each turn, enabling coherent, context-aware multi-turn dialogues.

---

## 🗂️ Knowledge Graph Ontology

### Node Types

| Label | Description |
|---|---|
| `PROCESS` | Core public service / procedure (central concept) |
| `TOPIC` | Thematic classification |
| `KEYWORD` | Search keyword associated with a process |
| `CONDITION` | Eligibility condition or requirement |
| `CONDITION_TYPE` | Category of condition |
| `EVIDENCE` | Required document or proof |
| `EVIDENCE_TYPE` | Category of evidence |
| `STEP` | Physical/manual process step |
| `STEP_DIGITAL` | Digital process step |
| `ORG` | Responsible or provision organization |

### Relationship Types

| Relationship | Meaning |
|---|---|
| `HAS_TOPIC` | Process belongs to topic |
| `HAS_KEYWORD` | Process tagged with keyword |
| `HAS_CONDITION` | Process requires condition |
| `HAS_EVIDENCE` | Process requires document |
| `HAS_STEP` | Process has manual step |
| `HAS_STEP_DIGITAL` | Process has digital step |
| `HAS_NEXT_STEP` | Sequential step ordering |
| `HAS_ORG_OWNER` | Owning organization |
| `HAS_PROVISION_ORG` | Service provision organization |
| `HAS_PARENT` | Hierarchical parent process |
| `HAS_RELATED` | Related processes |
| `HAS_SEM_SIMILAR_PROCESS` | Semantically similar process (precomputed) |
| `HAS_SEM_SIMILAR_CONDITION` | Semantically similar condition (precomputed) |
| `HAS_SEM_SIMILAR_EVIDENCE` | Semantically similar evidence (precomputed) |
| `HAS_SEM_SIMILAR_STEP` | Semantically similar step (precomputed) |

---

## 🗃️ Project Structure

```
.
├── app.py                  # Streamlit UI — chat interface + D3.js graph visualization
├── pipeline.py             # End-to-end answer_query_stream() orchestrator
├── query_understanding.py  # Greek normalization, intent classification, entity extraction
├── graph_traversal.py      # Ontology-aware Neo4j traversal by relationship type
├── retrieval.py            # Hybrid retrieval: vector + fulltext + graph
├── reranker.py             # Cross-encoder reranking
├── llm_response.py         # LLM response generation + ConversationMemory
├── neo4j_connector.py      # Neo4j connection pooling
├── cache.py                # Query and embedding caching
├── config.py               # Centralized configuration
├── setup_indexes.cypher    # Neo4j indexes + constraints setup
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🛠️ Technologies

| Technology | Role |
|---|---|
| **Neo4j** | Graph database — stores the full MITOS ontology |
| **OpenAI** (`text-embedding-3-large`) | State-of-the-art embeddings for semantic search |
| **LangChain** | LLM orchestration and conversation memory |
| **Streamlit** | Frontend — chat UI and graph display |
| **D3.js** | Interactive knowledge graph visualization |
| **cross-encoder/ms-marco** | Reranking retrieved candidates |
| **APOC** | Neo4j plugin for fuzzy text matching |
| **Levenshtein** | Greek morphological fuzzy matching |
| **Python-dotenv** | Environment variable management |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Neo4j instance (local or AuraDB) with MITOS data loaded
- OpenAI API key

### Installation

```bash
git clone https://github.com/yourusername/mitos-graph-rag
cd mitos-graph-rag
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_key
```

### Set Up Neo4j Indexes

```bash
cypher-shell -f setup_indexes.cypher
```

### Run the App

```bash
streamlit run app.py
```

---

## 📊 Performance Improvements vs. Original

| Dimension | Original | Refactored |
|---|---|---|
| Greek query matching | Exact `CONTAINS` only | Fuzzy + normalization + semantic edges |
| Score fusion | Raw score UNION (incomparable) | RRF normalization |
| Retrieval precision | None | Cross-encoder reranking |
| Graph traversal | Generic subgraph expansion | Intent-driven, relationship-typed |
| Embedding model | `text-embedding-ada-002` | `text-embedding-3-large` |
| Response streaming | ❌ | ✅ |
| Conversation memory | ❌ | ✅ |
| Source citations | ❌ | ✅ |
| Graph visualization | PyVis | D3.js (custom, interactive) |
| Query caching | ❌ | ✅ |

---

## 📄 License

MIT License — see `LICENSE` for details.
