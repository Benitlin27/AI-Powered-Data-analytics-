import json
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # Change to "mistral" or "phi3" if preferred


def _call_ollama(prompt: str) -> str:
    """Send a prompt to the local Ollama server and return the response text."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response from model.")
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Could not connect to Ollama. "
            "Make sure Ollama is running: open a terminal and run `ollama serve`, "
            "then pull a model with `ollama pull llama3`."
        )
    except Exception as e:
        return f"⚠️ LLM error: {str(e)}"


def get_llm_summary(aggregated: dict) -> str:
    """
    Ask Ollama to summarize and provide insights from the aggregated results.
    """
    prompt = f"""
You are an expert data scientist assistant. A user has uploaded a dataset and run an automated analysis pipeline.
Here are the results:

Dataset Shape: {aggregated['dataset_shape']['rows']} rows × {aggregated['dataset_shape']['cols']} columns
Columns: {', '.join(aggregated['columns'][:15])}
Total Missing Values: {aggregated['total_missing_values']}
Highly Skewed Features: {aggregated['highly_skewed_features']}

Top Feature Correlations:
{json.dumps(aggregated['top_correlations'], indent=2)}

ML Task Detected: {aggregated['task_type']}
Best Model: {aggregated['best_model']}

Model Performance Scores:
{json.dumps(aggregated['model_scores'], indent=2)}

Top Important Features:
{json.dumps(aggregated['top_features'], indent=2)}

Please provide:
1. A brief plain-English summary of the dataset (2-3 sentences)
2. Key findings from the EDA (3 bullet points)
3. Model performance analysis (2-3 sentences)
4. Top 3 actionable recommendations for improving the model or data quality
5. Any warnings or concerns about the data

Keep the response clear, concise, and non-technical enough for a business user.
"""
    return _call_ollama(prompt)


def ask_followup(question: str, aggregated: dict, chat_history: list) -> str:
    """
    Answer a follow-up question about the data using the aggregated context.
    Includes recent chat history for multi-turn conversation.
    """
    # Build recent conversation context (last 6 messages)
    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    context_summary = f"""
Dataset: {aggregated['dataset_shape']['rows']} rows × {aggregated['dataset_shape']['cols']} cols
Columns: {', '.join(aggregated['columns'][:15])}
Task: {aggregated['task_type']}
Best Model: {aggregated['best_model']}
Top Features: {[f['feature'] for f in aggregated['top_features']]}
Model Scores: {json.dumps(aggregated['model_scores'], indent=2)}
Top Correlations: {json.dumps(aggregated['top_correlations'][:3], indent=2)}
"""

    prompt = f"""
You are an expert data scientist answering questions about a dataset analysis.

Dataset Context:
{context_summary}

Recent Conversation:
{history_text}

User Question: {question}

Answer clearly and concisely. If the answer requires looking at the data more closely,
say what additional analysis would help. Keep the response under 200 words.
"""
    return _call_ollama(prompt)
