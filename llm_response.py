# llm_response.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OpenAI API Key. Set it in the .env file.")


llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)

def generate_response(user_query, graph_data, mode="text_only"):
    """
    Generates a structured Greek response based on knowledge graph data.
    mode: "text_only" or "hybrid"
    """

    if mode == "text_only":
        # Old structure: node_1, relationship, node_2
        context = "\n".join([
            f"➤ **{item['node_1']}** → {item['relationship']} → **{item['node_2']}**"
            for item in graph_data
        ])
    else:
        # Hybrid structure includes a 'score'
        context = "\n".join([
            f"➤ **{item['node_1']}** (Score: {item['score']:.2f}) → {item['relationship']} → **{item['node_2']}**"
            for item in graph_data
        ])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Είσαι ένας εξειδικευμένος ψηφιακός βοηθός για τον **Εθνικό Κατάλογο Υπηρεσιών της Ελλάδας**. 
        Σου δίνω πληροφορίες από τη Γραφική Βάση Γνώσης (GraphRAG) και θέλω να απαντήσεις με **δομημένο και επαγγελματικό τρόπο στα ελληνικά**.

        🔹 **Ερώτηση:** {question}
        🔹 **Σχετικές Πληροφορίες:**
        {context}

        🔹 **Απάντηση:** Δώσε μια αναλυτική, σαφή και δομημένη απάντηση στα ελληνικά. Εξήγησε τη διαδικασία, τα βήματα, και αν υπάρχουν σχετικές υπηρεσίες.
        """
    )

    formatted_prompt = prompt.format(context=context, question=user_query)
    response = llm.predict(formatted_prompt)
    return response
