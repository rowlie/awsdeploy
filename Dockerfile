import os
import json
import torch
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from datetime import datetime
from typing import List, Dict, Any

# --- Configuration (Pulled from AWS App Runner Environment Variables) ---
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "youtube-qa-index")
TOP_K = 5
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Check for essential keys before proceeding
if not PINECONE_KEY or not OPENAI_API_KEY:
    # App Runner will report this crash in the Deployment Logs
    raise ValueError("One or more required environment variables (PINECONE_API_KEY, OPENAI_API_KEY) are not set.")


# --- Tool Definitions ---

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Example: '2 + 2 * 5' or '10 / 3'
    """
    try:
        # NOTE: eval() is used for simplicity but requires care in production
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def word_count(text: str) -> str:
    """Count the number of words in a given text."""
    count = len(text.split())
    return f"Word count: {count}"

@tool
def convert_case(text: str, case_type: str) -> str:
    """
    Convert text to uppercase, lowercase, or title case.
    case_type options: 'upper', 'lower', 'title'
    """
    if case_type == "upper":
        return text.upper()
    elif case_type == "lower":
        return text.lower()
    elif case_type == "title":
        return text.title()
    else:
        return f"Error: Unknown case type '{case_type}'. Use 'upper', 'lower', or 'title'."

TOOLS = [calculator, get_current_time, word_count, convert_case]


# --- RAG and LLM Initialization ---

def initialize_components():
    """Initialize Sentence Transformer, Pinecone, and LangChain LLM."""
    print("Initializing RAG and LLM components...")
    
    # --- CRITICAL FIX: Force CPU to avoid container startup crash ---
    device = "cpu" # Changed from auto-detect to explicitly use CPU
    
    # Retriever model
    retriever = SentenceTransformer(
        "flax-sentence-embeddings/all_datasets_v3_mpnet-base",
        device=device,
    )

    # Pinecone client + index
    pc = Pinecone(api_key=PINECONE_KEY)
    index = pc.Index(INDEX_NAME)
    
    # LangChain LLM wrappers
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
    llm_with_tools = llm.bind_tools(TOOLS)

    print(f"✅ Components initialized. Pinecone Index: {INDEX_NAME}. LLM: gpt-3.5-turbo")
    return retriever, index, llm, llm_with_tools

# Global initialization (Runs once when the Flask app starts)
try:
    RETRIEVER, INDEX, LLM, LLM_WITH_TOOLS = initialize_components()
except Exception as e:
    print(f"CRITICAL ERROR during initialization: {e}")
    # Propagate exception to ensure container exits gracefully
    raise e


# --- RAG Helper Functions ---

def retrieve_pinecone_context(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """Query Pinecone with an embedded version of the user query."""
    # Use the globally loaded RETRIEVER
    xq = RETRIEVER.encode(query).tolist()
    # Use the globally loaded INDEX
    res = INDEX.query(vector=xq, top_k=top_k, include_metadata=True)
    return res

def context_string_from_matches(matches: List[Dict[str, Any]]) -> str:
    """Build a single context string from Pinecone matches."""
    parts = []
    for m in matches:
        meta = m["metadata"]
        passage = meta.get("text") or meta.get("passage_text") or ""
        if passage:
            parts.append(passage)
    return "\n\n".join(parts)


# --- LangChain Runnable Chain Steps ---

# STEP 1: Runnable to build messages + RAG context
def _build_messages(inputs: dict) -> dict:
    user_message = inputs["user_message"]
    history_messages = inputs["history"]

    # RAG: retrieve context from Pinecone using only the latest user message
    pinecone_result = retrieve_pinecone_context(user_message)
    context = context_string_from_matches(pinecone_result.get("matches", []))

    # Combine history + current user message + RAG context
    messages = list(history_messages)
    messages.append(HumanMessage(content=user_message))

    if context:
        messages.append(
            HumanMessage(
                content=f"📚 Relevant context from knowledge base for this query:\\n{context}"
            )
        )

    return {
        "messages": messages,
        "rag_context": context,
    }

# STEP 2: Runnable to call the tool-enabled LLM (decision step)
def _first_llm_call(state: dict) -> dict:
    messages = state["messages"]
    # Uses the global LLM with bound tools
    first_response = LLM_WITH_TOOLS.invoke(messages)
    return {
        **state,
        "first_response": first_response,
    }

# STEP 3: Runnable to execute any tools requested by the first LLM call
def _run_tools_if_needed(state: dict) -> dict:
    first_response = state["first_response"]
    messages = state["messages"]

    tool_calls = getattr(first_response, "tool_calls", None)
    if not tool_calls and hasattr(first_response, "additional_kwargs"):
        tool_calls = first_response.additional_kwargs.get("tool_calls")

    if not tool_calls:
        # No tools needed, prepare message list for final LLM call
        return {
            **state,
            "messages_with_tools": messages + [first_response],
        }

    tool_results_messages = []
    for call in tool_calls:
        tool_name = call.get("name") or call.get("function", {}).get("name")
        raw_args = (
            call.get("args")
            or call.get("arguments")
            or call.get("function", {}).get("arguments", {})
        )

        if isinstance(raw_args, str):
            try:
                tool_args = json.loads(raw_args)
            except Exception:
                tool_args = {}
        else:
            tool_args = raw_args or {}

        tool_id = call.get("id", "tool_call_id_default") 
        matching = [t for t in TOOLS if t.name == tool_name]

        if not matching:
            result_text = f"Tool '{tool_name}' not found."
        else:
            try:
                # Invoke the tool
                result_text = matching[0].invoke(tool_args)
            except Exception as e:
                result_text = f"Error in tool '{tool_name}': {e}"
        
        tool_results_messages.append(
            ToolMessage(
                content=str(result_text),
                tool_call_id=tool_id,
            )
        )

    # Build new message list including first_response + tool outputs
    messages_with_tools = messages + [first_response] + tool_results_messages

    return {
        **state,
        "messages_with_tools": messages_with_tools,
    }

# STEP 4: Runnable to call plain LLM for final answer
def _final_llm_call(state: dict) -> dict:
    messages_with_tools = state["messages_with_tools"]
    # Final call uses the global plain LLM (without forcing tools)
    final_response = LLM.invoke(messages_with_tools)
    return {
        "final_response": final_response,
        "rag_context": state.get("rag_context", ""),
    }

# Compose the Runnable chain
RAG_AGENT_CHAIN = (
    RunnableLambda(lambda inputs: {"user_message": inputs["user_message"], "history": inputs["history"]})
    | RunnableLambda(_build_messages)
    | RunnableLambda(_first_llm_call)
    | RunnableLambda(_run_tools_if_needed)
    | RunnableLambda(_final_llm_call)
)


# --- Flask Application Setup ---

app = Flask(__name__)

# Basic health check for App Runner
@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """App Runner requires a healthy endpoint."""
    # This check confirms the Flask app is running.
    return jsonify({"status": "healthy", "service": INDEX_NAME}), 200

@app.route('/chat', methods=['POST'])
def chat():
    """
    API endpoint for the RAG chat.
    Expects JSON body: {"message": str, "history": List[Dict]}
    Returns JSON body: {"response": str}
    """
    try:
        data = request.get_json()
        user_message = data.get('message')
        raw_history = data.get('history', [])

        if not user_message:
            return jsonify({"error": "Missing 'message' in request body"}), 400

        # Convert raw history (from frontend) into LangChain BaseMessage objects
        history_messages: List[BaseMessage] = []
        for msg in raw_history:
            if msg.get("type") == "human":
                history_messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("type") == "ai":
                history_messages.append(AIMessage(content=msg.get("content", "")))

        # Invoke the RAG chain
        chain_inputs = {
            "user_message": user_message,
            "history": history_messages
        }
        
        result = RAG_AGENT_CHAIN.invoke(chain_inputs)
        final_response_content = result["final_response"].content

        return jsonify({
            "response": final_response_content,
        }), 200

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Log the full exception for debugging in AWS CloudWatch
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    # Used only for local debugging (App Runner uses gunicorn)
    print("Running Flask app locally...")
    app.run(debug=True, host='0.0.0.0', port=os.environ.get("PORT", 8080))
