import chromadb
from flask import Flask, request, jsonify, render_template
from groq import Groq
import traceback
from dotenv import load_dotenv
import os

# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()

# -------------------------
# GROQ API KEY (from .env)
# -------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------
# ChromaDB
# -------------------------
chroma_client = chromadb.PersistentClient(path="vectorstore")

collection = chroma_client.get_or_create_collection(
    name="vettri_ai"
)

print("Vector DB Loaded:", collection.count())

# -------------------------
# Flask
# -------------------------
app = Flask(__name__)

# -------------------------
# Chat Memory (Session Level)
# -------------------------
chat_history = []

# -------------------------
# Retrieve Context
# -------------------------
def retrieve_context(question):
    results = collection.query(
        query_texts=[question],
        n_results=5
    )
    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""
    return "\n".join(docs)

# -------------------------
# Generate Response
# -------------------------
def generate_response(question):
    global chat_history

    context = retrieve_context(question)

    # Add user message
    chat_history.append({"role": "user", "content": question})

    system_prompt = f"""
You are Vettri AI, a friendly and helpful assistant for students.

Guidelines:
- Answer clearly using the given context
- Keep responses natural and conversational
- Do not be overly emotional
- If answer is not in context, say you are not sure

CONTEXT:
{context}
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                *chat_history
            ],
            temperature=0.7,
            max_completion_tokens=512,
            top_p=1
        )

        answer = completion.choices[0].message.content

        if not answer:
            return "I'm having a little trouble responding right now. Try again?"

        print("MODEL RESPONSE:", answer)

        # Save assistant reply
        chat_history.append({"role": "assistant", "content": answer})

        return answer

    except Exception as e:
        print("GROQ ERROR:", type(e).__name__, str(e))
        traceback.print_exc()
        return "⚠️ Vettri AI is temporarily unavailable."

# -------------------------
# Homepage
# -------------------------
@app.route("/")
def home():
    return render_template("Achievia.html")

# -------------------------
# Ask Endpoint
# -------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"answer": "Please enter a question."})

    answer = generate_response(question)
    return jsonify({"answer": answer})

# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
