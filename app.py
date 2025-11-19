import dotenv
import weaviate
from weaviate.classes.init import Auth
from langchain_community.embeddings import HuggingFaceEmbeddings
from weaviate.classes.query import MetadataQuery 
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import re
import google.generativeai as genai



dotenv.load_dotenv()
API_KEY = dotenv.get_key(dotenv.find_dotenv(), "GEMINI_API")
WEAVIATE_URL = dotenv.get_key(dotenv.find_dotenv(), "WEAVIATE_URL")
WEAVIATE_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "WEAVIATE_API_KEY")

genai.configure(api_key=API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

app = Flask(__name__)
CORS(app)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def retrieve_relevant_chunks(query: str, top_k=5):
    vector = embeddings.embed_query(query)
    collection = client.collections.get("PDFChunk")
    response = collection.query.near_vector(
        near_vector=vector,
        limit=top_k,
        return_metadata=MetadataQuery(distance=True)
    )
    print("\nRetrieved Chunks:")
    print(response)
    return response




def generate_answer_from_retrieved_chunks(query: str, results) -> str:
    """
    Generates an answer using the Gemini model based on retrieved chunks from Weaviate.
    """
    # 1. Extract content from Weaviate results
    retrieved_chunks = [obj.properties["content"] for obj in results.objects]
    context = "\n\n".join(retrieved_chunks) if retrieved_chunks else "No relevant context found."

    print("\nContext for Gemini:")
    print(context)

    # 2. Build the prompt properly
    prompt_text = f"""
        You are an AI assistant answering based on the following context.

        Context:
        {context}

        Question:
        {query}

        Provide a concise answer based ONLY on the context above.
        If the context does not contain the answer, say "I don't know."
    """

    # 3. Call Gemini correctly
    response = gemini.generate_content(prompt_text)

    # 4. Return answer
    return response.text.strip()






@app.route('/rag', methods=['POST'])
def rag_endpoint():
    if not client.is_ready():
        return jsonify({"error": "Weaviate is not ready"}), 503
    
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    results = retrieve_relevant_chunks(query)
    answer = generate_answer_from_retrieved_chunks(query, results)
    print("\nAnswer:\n", answer)
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)


    return jsonify({
        "query": query,
        "answer": answer,
    })


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,          
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True
    )

    try:
        app.run(debug=True, port=5000)
    finally:
        client.close()
        print("✅ Client connection closed.")


