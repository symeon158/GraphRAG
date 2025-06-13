# populate_chroma.py

from simple_rag import populate_chroma_from_csv

# Path to the file you prepared with 'combined_text'
csv_path = r"C:\Users\sisma\OneDrive\Υπολογιστής\Mitos_Data\mitos_context_ready.csv"

populate_chroma_from_csv(csv_path)
