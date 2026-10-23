
import os
import time
import json
import re
import pickle

import faiss
from flask import Flask, render_template, request, jsonify
from google import genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# ==========================================
# 1. INITIALIZATION
# ==========================================

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Please check your .env file."
    )

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.5-flash-lite"


# ==========================================
# 2. LOAD RAG KNOWLEDGE BASE
# ==========================================

print("Loading RAG knowledge base...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("knowledge_base/faiss_index.bin")

with open("knowledge_base/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)

print(f"RAG knowledge base loaded: {index.ntotal} vectors")


# ==========================================
# 3. SHARED GEMINI HELPER
# ==========================================

def generate_ai_response(prompt, max_retries=3):
    """
    Generate a Gemini response with limited retries for
    temporary service errors and rate limiting.

    API quota exhaustion is not retried repeatedly.
    """

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            answer = response.text

            if not answer or not answer.strip():
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return answer.strip()

        except Exception as error:
            error_message = str(error).upper()

            is_quota_exhausted = (
                "RESOURCE_EXHAUSTED" in error_message
                or "QUOTA" in error_message
                or "INSUFFICIENT_QUOTA" in error_message
            )

            is_temporary = (
                "503" in error_message
                or "UNAVAILABLE" in error_message
                or "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
                or "429 TOO MANY REQUESTS" in error_message
            )

            # Do not keep retrying when the account quota is exhausted.
            if is_quota_exhausted:
                print("Gemini quota/rate-limit error:", error)
                raise RuntimeError(
                    "Gemini API quota or rate limit reached. "
                    "Check your Gemini API quota and billing."
                ) from error

            # Retry only errors that appear temporary.
            if is_temporary and attempt < max_retries - 1:
                wait_seconds = 2 ** attempt

                print(
                    "Gemini temporarily unavailable. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)
                continue

            print("Gemini generation error:", error)

            if is_temporary:
                raise RuntimeError(
                    "Gemini is temporarily unavailable after "
                    "multiple attempts. Please try again later."
                ) from error

            raise RuntimeError(
                "Gemini could not generate a response. "
                "Check the server logs for details."
            ) from error


def ai_error_response(error, feature):
    """Return a clear error response for an AI-powered feature."""

    message = str(error)

    if "quota or rate limit" in message.lower():
        status_code = 429
    elif "temporarily unavailable" in message.lower():
        status_code = 503
    else:
        status_code = 500

    print(f"{feature} error: {error}")

    return jsonify({"error": message}), status_code


# ==========================================
# 4. HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# 5. ASK OSTUTOR — RAG
# ==========================================

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")

    if not isinstance(question, str):
        return jsonify({
            "answer": "Please enter a valid question."
        }), 400

    question = question.strip()

    if not question:
        return jsonify({
            "answer": "Please enter an Operating Systems question."
        }), 400

    try:
        # Step 1: Create the question embedding.
        question_embedding = embedding_model.encode(
            [question],
            convert_to_numpy=True
        )

        # Step 2: Search FAISS for relevant chunks.
        k = min(3, index.ntotal)

        if k == 0:
            return jsonify({
                "answer": "The knowledge base contains no indexed material."
            }), 500

        distances, indices = index.search(question_embedding, k)

        # Step 3: Retrieve relevant lecture material.
        retrieved_chunks = []

        for chunk_index in indices[0]:
            if 0 <= chunk_index < len(chunks):
                retrieved_chunks.append(chunks[chunk_index])

        if not retrieved_chunks:
            return jsonify({
                "answer": (
                    "No relevant lecture material was found. "
                    "Please try rephrasing your question."
                ),
                "sources": []
            }), 200

        # Step 4: Build context and source references.
        context_parts = []
        sources = []

        for chunk in retrieved_chunks:
            if not isinstance(chunk, dict):
                continue

            chunk_text = chunk.get("text", "")
            if chunk_text:
                context_parts.append(chunk_text)

            source_name = chunk.get("source", "Unknown source")
            location = chunk.get("location", "Unknown location")
            source_label = f"{source_name} - {location}"

            if source_label not in sources:
                sources.append(source_label)

        if not context_parts:
            return jsonify({
                "answer": "No readable lecture text was retrieved.",
                "sources": sources
            }), 200

        context = "\n\n".join(context_parts)

        # Step 5: Build the grounded RAG prompt.
        prompt = f"""
You are OSTutorLLM, an AI tutor specialized in Operating Systems.

Answer the student's question using the retrieved lecture material.

Rules:
- Use the retrieved lecture material as the primary source.
- Explain concepts clearly in simple language.
- Use correct Operating Systems terminology.
- Do not invent facts or claim that unsupported information
  came from the lecture material.
- If the material is insufficient, state that clearly.
- You may provide general OS knowledge when useful, but
  distinguish it from information supported by the material.
- Use examples or numbered steps when appropriate.

Retrieved lecture material:
{context}

Student question:
{question}
"""

        # Step 6: Generate the answer with retry handling.
        answer = generate_ai_response(prompt)

        # Step 7: Return answer and source references separately.
        return jsonify({
            "answer": answer,
            "sources": sources
        }), 200

    except Exception as error:
        response, status_code = ai_error_response(error, "Ask OS Tutor")

        return jsonify({
            "answer": response.get_json().get(
                "error",
                "An error occurred while generating your answer."
            )
        }), status_code


# ==========================================
# 6. MCQ PRACTICE
# ==========================================

@app.route("/mcq", methods=["POST"])
def mcq():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "Operating Systems")

    if not isinstance(topic, str) or not topic.strip():
        topic = "Operating Systems"

    topic = topic.strip()[:200]

    prompt = f"""
Create 5 multiple-choice questions about {topic}
for an Operating Systems student.

Return valid JSON only in this exact structure:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": [
        "A. Option",
        "B. Option",
        "C. Option",
        "D. Option"
      ],
      "answer": "A",
      "explanation": "Brief explanation"
    }}
  ]
}}

Requirements:
- Create exactly 5 questions.
- Use accurate Operating Systems concepts.
- Include exactly four options per question.
- The answer must be A, B, C, or D.
- Ensure the indicated answer matches the correct option.
- Do not include Markdown fences or extra text.
"""

    try:
        response_text = generate_ai_response(prompt)

        # Handle responses that include Markdown code fences.
        cleaned_text = re.sub(
            r"^\s*```(?:json)?\s*|\s*```\s*$",
            "",
            response_text,
            flags=re.IGNORECASE
        )

        result = json.loads(cleaned_text)

        questions = result.get("questions")

        if not isinstance(questions, list) or not questions:
            raise ValueError("Gemini returned an invalid MCQ structure.")

        valid_questions = []

        for question in questions:
            if not isinstance(question, dict):
                continue

            options = question.get("options")
            answer = str(question.get("answer", "")).strip().upper()

            if (
                isinstance(question.get("question"), str)
                and isinstance(options, list)
                and len(options) == 4
                and answer in {"A", "B", "C", "D"}
            ):
                question["answer"] = answer
                valid_questions.append(question)

        if not valid_questions:
            raise ValueError("No valid MCQs were returned.")

        result["questions"] = valid_questions[:5]

        return jsonify(result), 200

    except Exception as error:
        return ai_error_response(error, "MCQ Practice")


# ==========================================
# 7. VIVA PRACTICE
# ==========================================

@app.route("/viva", methods=["POST"])
def viva():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "Operating Systems")

    if not isinstance(topic, str) or not topic.strip():
        topic = "Operating Systems"

    topic = topic.strip()[:200]

    prompt = f"""
Act as an Operating Systems viva examiner.

Generate one clear viva question about: {topic}.

Return only the question, without an answer.
"""

    try:
        question = generate_ai_response(prompt)

        return jsonify({"question": question}), 200

    except Exception as error:
        return ai_error_response(error, "Viva Practice")


# ==========================================
# 8. VIVA ANSWER EVALUATION
# ==========================================

@app.route("/viva-feedback", methods=["POST"])
def viva_feedback():
    data = request.get_json(silent=True) or {}

    question = data.get("question", "")
    answer = data.get("answer", "")

    if not isinstance(question, str) or not isinstance(answer, str):
        return jsonify({
            "error": "The question and answer must be text."
        }), 400

    question = question.strip()
    answer = answer.strip()

    if not question or not answer:
        return jsonify({
            "error": "A question and your answer are required."
        }), 400

    prompt = f"""
You are an Operating Systems viva examiner.

Evaluate the student's answer fairly and constructively.

Question:
{question}

Student answer:
{answer}

Provide:
1. What is correct
2. What is missing or incorrect
3. A suggested model answer
4. A score out of 10

Use clear headings and explain the score briefly.
Do not claim that an answer is correct if it is not.
"""

    try:
        feedback = generate_ai_response(prompt)

        return jsonify({"feedback": feedback}), 200

    except Exception as error:
        return ai_error_response(error, "Viva Feedback")


# ==========================================
# 9. LINUX AND SHELL TUTOR
# ==========================================

@app.route("/linux", methods=["POST"])
def linux():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")

    if not isinstance(question, str):
        return jsonify({
            "error": "Enter a valid Linux or Shell question."
        }), 400

    question = question.strip()

    if not question:
        return jsonify({
            "error": "Enter a Linux or Shell question."
        }), 400

    question = question[:3000]

    prompt = f"""
You are a Linux and Shell programming tutor.

Answer the user's question clearly and accurately.

Requirements:
- Explain the command or script.
- Give a safe example when useful.
- Describe expected output when appropriate.
- Explain shell scripts step by step when requested.
- Warn the user before suggesting destructive commands.
- Do not claim that commands have been executed.

User question:
{question}
"""

    try:
        answer = generate_ai_response(prompt)

        return jsonify({"answer": answer}), 200

    except Exception as error:
        return ai_error_response(error, "Linux Tutor")


# ==========================================
# 10. START FLASK SERVER
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
