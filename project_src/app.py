# app.py

import os
import json
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# -----------------------
# CONFIG
# -----------------------

# Candidate chat models (instruction-tuned, HF IDs)
MODEL_CONFIGS = {
    "Llama 3 8B Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "Gemma 2 9B IT": "google/gemma-2-9b-it",
    "Gemma 2 2B IT": "google/gemma-2-2b-it",
    "Llama 3.2 1B Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "SmolLM2 135M Instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
}

# Ordered list of (smaller) fallback models to try if the selected one fails
FALLBACK_ORDER = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]

MAX_TURNS_MEMORY = 6          # short-term memory (last N user/assistant pairs)
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, honest, and safe conversational AI assistant. 
- Be clear and concise.
- If you are unsure, say so and suggest how the user might clarify.
- Follow user instructions carefully and prioritize user intent.
- Never pretend to have capabilities you do not have.
""".strip()

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "conversations.jsonl")

# -----------------------
# GLOBAL MODEL CACHE
# -----------------------

current_tokenizer = None
current_model = None
current_model_id = None


def load_model(preferred_model_id: str, progress=None):
    """
    Try to load the preferred model; if it fails (e.g. GPU OOM),
    fall back through FALLBACK_ORDER.
    """
    global current_model, current_tokenizer, current_model_id

    if current_model_id == preferred_model_id and current_model is not None:
        # Already loaded
        if progress:
            progress(1.0, desc=f"Model {preferred_model_id} already loaded.")
        return current_model_id

    # Build candidate list: preferred first, then fallbacks (no duplicates)
    candidates = [preferred_model_id] + [
        mid for mid in FALLBACK_ORDER if mid != preferred_model_id
    ]

    last_error = None
    for i, model_id in enumerate(candidates):
        try:
            msg = f"Loading model: {model_id}"
            print(msg)
            if progress:
                progress((i / len(candidates)) * 0.8, desc=msg)
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Determine the best available device and dtype
            if torch.cuda.is_available():
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
            )

            current_model = model
            current_tokenizer = tokenizer
            current_model_id = model_id
            print(f"Loaded model: {model_id}")
            return model_id

        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            print(f"OOM loading {model_id}, trying next fallback...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            last_error = e
            print(f"Error loading {model_id}: {e}, trying next fallback...")

    raise RuntimeError(f"Could not load any model. Last error: {last_error}")


# -----------------------
# PROMPT / GENERATION LOGIC
# -----------------------

def build_prompt(system_prompt, history, user_message):
    """
    Use a short-term memory buffer. Converts history into a list of messages
    and uses tokenizer.apply_chat_template if available.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add last N turns
    # History is list of dicts: [{'role': 'user', 'content': ...}, ...]
    start_idx = max(0, len(history) - (MAX_TURNS_MEMORY * 2))
    messages.extend(history[start_idx:])

    # Append new user message
    messages.append({"role": "user", "content": user_message})

    # Prefer chat template if provided by tokenizer
    if hasattr(current_tokenizer, "apply_chat_template"):
        try:
            prompt = current_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        except Exception as e:
            print("apply_chat_template failed, falling back to plain format:", e)

    # Fallback: simple role-tagged text
    formatted = ""
    for m in messages:
        formatted += f"{m['role'].upper()}: {m['content']}\n"
    formatted += "ASSISTANT:"
    return formatted


def generate_response(
    user_message,
    history,
    system_prompt,
    temperature,
    max_new_tokens,
    model_choice,
    progress=gr.Progress(),
):
    """
    Core chat function used by Gradio.
    Returns updated history, status text, and an interaction_id for logging feedback.
    """
    if not user_message.strip():
        return history, "Please type a message.", None

    # Show a system-level loading message
    loading_notice = "Loading model and thinking... (Cold starts may take a bit on free GPUs.)"
    progress(0, desc="Initializing...")

    # Resolve model ID from UI choice, then ensure a model is loaded (with fallback)
    preferred_model_id = MODEL_CONFIGS.get(model_choice, list(MODEL_CONFIGS.values())[0])

    try:
        active_model_id = load_model(preferred_model_id, progress=progress)
    except Exception as e:
        # Hard failure: tell user and stop
        status = f"Error loading models: {e}"
        history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": f"[SYSTEM]: {status}"}]
        return history, status, None

    # Build prompt with short-term memory
    progress(0.8, desc="Building prompt...")
    prompt = build_prompt(system_prompt or DEFAULT_SYSTEM_PROMPT, history, user_message)

    device = next(current_model.parameters()).device
    inputs = current_tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    # Generate
    progress(0.9, desc="Generating response...")
    with torch.no_grad():
        output_ids = current_model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            pad_token_id=current_tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    assistant_text = current_tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    # Update history (Gradio Chatbot expects list of dicts)
    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_text}
    ]

    # Logging
    interaction_id = log_interaction(
        user_message=user_message,
        assistant_text=assistant_text,
        history=new_history,
        model_id=active_model_id,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    status = f"Model: {active_model_id} | Temperature: {temperature} | Max tokens: {max_new_tokens}"
    return new_history, status, interaction_id


# -----------------------
# LOGGING
# -----------------------

def log_interaction(
    user_message,
    assistant_text,
    history,
    model_id,
    temperature,
    max_new_tokens,
):
    """
    Append a JSON line with interaction metadata.
    """
    interaction_id = f"{int(time.time() * 1000)}"
    record = {
        "interaction_id": interaction_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_id": model_id,
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "user_message": user_message,
        "assistant_text": assistant_text,
        "history": history,
        "helpful": None,  # to be updated by feedback
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print("Error writing log:", e)

    return interaction_id


def record_feedback(feedback, interaction_id):
    """
    Super-simple feedback logger.
    We don't rewrite old lines (too complex for demo); we just append another record.
    """
    if not interaction_id:
        return "No interaction to attach feedback to."
    if feedback not in ["👍", "👎"]:
        return "Please select 👍 or 👎 first."

    record = {
        "interaction_id": interaction_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "feedback": feedback,
    }
    try:
        with open(os.path.join(LOG_DIR, "feedback.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print("Error writing feedback log:", e)

    return "Thanks for your feedback!"


# -----------------------
# GRADIO UI
# -----------------------

EXAMPLE_PROMPTS = [
    "Explain the main ideas behind human–computer interaction in simple terms.",
    "Help me design a user study to evaluate a chatbot interface.",
    "Rewrite this chatbot error message so it’s friendlier to users.",
    "List potential usability problems with a long conversational context window.",
]


def clear_history():
    return [], "", None


with gr.Blocks(title="HCI-Focused Conversational AI Chatbot") as demo:
    gr.Markdown(
        """
        # HCI-Focused Conversational AI Chatbot
        
        This chatbot runs on open-source LLMs hosted on HuggingFace.
        It supports short-term conversational memory, adjustable answer length,
        and temperature for controllability.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=450,
            )
            status_box = gr.Markdown("Ready.")
        with gr.Column(scale=2):
            system_prompt_box = gr.Textbox(
                label="System Prompt (optional)",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=8,
            )
            model_selector = gr.Dropdown(
                label="Preferred Model",
                choices=list(MODEL_CONFIGS.keys()),
                value="Llama 3 8B Instruct",
            )
            temperature_slider = gr.Slider(
                label="Temperature (creativity)",
                minimum=0.0,
                maximum=1.5,
                step=0.05,
                value=0.7,
            )
            max_tokens_slider = gr.Slider(
                label="Answer Length (max new tokens)",
                minimum=64,
                maximum=1024,
                step=32,
                value=256,
            )
            gr.Markdown("### Example Prompts")
            gr.Examples(
                examples=[[p] for p in EXAMPLE_PROMPTS],
                inputs=[],
                outputs=[],
                label=None,
            )
            gr.Markdown(
                """
                _Note: On free GPUs, there may be short cold-start delays or occasional
                model fallback to keep the conversation responsive._
                """
            )

    with gr.Row():
        user_input = gr.Textbox(
            label="Type your message",
            placeholder="Ask me anything...",
            lines=3,
        )

    with gr.Row():
        send_button = gr.Button("Send", variant="primary")
        clear_button = gr.Button("Clear Conversation")

    # Feedback row
    with gr.Row():
        feedback_radio = gr.Radio(
            ["👍", "👎"],
            label="Was this helpful?",
            info="Quick feedback helps improve future versions.",
        )
        feedback_button = gr.Button("Submit Feedback")
        feedback_status = gr.Markdown("")

    # Hidden state: last interaction_id for feedback
    interaction_state = gr.State(value=None)

    # Hooks
    send_button.click(
        fn=generate_response,
        inputs=[
            user_input,
            chatbot,
            system_prompt_box,
            temperature_slider,
            max_tokens_slider,
            model_selector,
        ],
        outputs=[chatbot, status_box, interaction_state],
    )

    # Also allow pressing Enter to send
    user_input.submit(
        fn=generate_response,
        inputs=[
            user_input,
            chatbot,
            system_prompt_box,
            temperature_slider,
            max_tokens_slider,
            model_selector,
        ],
        outputs=[chatbot, status_box, interaction_state],
    )

    clear_button.click(
        fn=clear_history,
        inputs=[],
        outputs=[chatbot, status_box, interaction_state],
    )

    feedback_button.click(
        fn=record_feedback,
        inputs=[feedback_radio, interaction_state],
        outputs=feedback_status,
    )

if __name__ == "__main__":
    # On Spaces you typically just call demo.launch(), but this keeps it runnable locally too.
    # server_name="0.0.0.0" allows external access (needed for Docker)
    demo.launch(server_name="0.0.0.0")