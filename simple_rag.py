# simple_rag.py

import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OpenAI API Key in the .env file.")

# Paths and setup
DATA_DIR = r"C:\Users\sisma\OneDrive\Υπολογιστής\Mitos_Data"
CHROMA_DB_PATH = "./chromadb_db"

# Load CSV files into DataFrames
tables = [
    "services.csv",
    "conditions.csv",
    "evidences.csv",
    "rules.csv",
    "steps.csv",
    "digital_steps.csv",
    "useful_links.csv",
    "provision_digital_locations.csv"
]

dataframes = {name.split('.')[0]: pd.read_csv(os.path.join(DATA_DIR, name), encoding='utf-8-sig')
              for name in tables}

# Select columns for embedding
columns_to_embed = {
    'services': ['official_title', 'description', 'org_owner_title_el', 'url'],
    'conditions': ['conditions_type', 'conditions_name', 'conditions_url'],
    'evidences': ['evidence_type_el', 'evidence_description', 'evidence_note'],
    'rules': ['rule_type', 'rule_description', 'rule_url'],
    'steps': ['step_title', 'step_description', 'step_note'],
    'digital_steps': ['step_digital_title', 'step_digital_implementation', 'step_digital_url'],
    'useful_links': ['useful_link_title', 'useful_link_url'],
    'provision_digital_locations': ['provision_digital_location_title', 'provision_digital_location_url']
}

# Combine textual data per service_id
def prepare_combined_text(dataframes, columns_to_embed):
    combined_text_per_service = {}

    for service_id in dataframes['services']['service_id'].unique():
        texts = []
        for table, cols in columns_to_embed.items():
            df = dataframes[table]
            relevant_rows = df[df['service_id'] == service_id]
            for _, row in relevant_rows.iterrows():
                row_text = " | ".join(row[col] for col in cols if pd.notnull(row[col]))
                texts.append(row_text)
        combined_text_per_service[service_id] = " ".join(texts)

    return combined_text_per_service

# ChromaDB client and embedding function
def init_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-ada-002"
    )
    collection = client.get_or_create_collection(
        name="mitos_simple_rag",
        embedding_function=openai_ef
    )
    return collection

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# Insert data into ChromaDB
def insert_embeddings(df_embeddings, collection):
    for idx, row in df_embeddings.iterrows():
        service_id = row['service_id']
        text = row['text']
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = f"{service_id}_chunk_{i}"
            # Avoid duplicate insertion
            try:
                collection.add(
                    documents=[chunk],
                    ids=[doc_id],
                    metadatas=[{"service_id": service_id}]
                )
            except Exception as e:
                print(f"Skipping duplicate or error for doc_id {doc_id}: {e}")

# Query function
def simple_rag_query(query, collection, top_k=5):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results['documents'][0]

# Main execution
if __name__ == "__main__":
    try:
        print("⏳ Starting data preparation and embedding...")
        
        combined_text_per_service = prepare_combined_text(dataframes, columns_to_embed)

        df_embeddings = pd.DataFrame({
            'service_id': list(combined_text_per_service.keys()),
            'text': list(combined_text_per_service.values())
        })

        collection = init_chroma_collection()

        insert_embeddings(df_embeddings, collection)
        print("✅ Embeddings successfully stored in ChromaDB.")

        # Test a query explicitly
        user_query = "Πώς μπορώ να εκδώσω πιστοποιητικό γέννησης;"
        print(f"\n🔍 Querying ChromaDB with: '{user_query}'")
        results = simple_rag_query(user_query, collection)
        print("\n🔖 Top Simple RAG Results:")
        for res in results:
            print("-", res, "\n")

    except Exception as e:
        print(f"❌ Error encountered: {e}")
