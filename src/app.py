# ---
# args: ["--message", "what's up?"]
# ---
"""Single-page application that lets you talk to a transformer chatbot.

This is a complex example demonstrating an end-to-end web application backed by
serverless web handlers and GPUs. The user visits a single-page application,
written using Solid.js. This interface makes API requests that are handled by a
Modal function running on the GPU.

The weights of the model are saved in the image, so they don't need to be
downloaded again while the app is running.

Chat history tensors are saved in a `modal.Dict` distributed dictionary.
"""

import uuid
from pathlib import Path
from typing import Optional, Tuple

import fastapi
import modal
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

assets_path = Path(__file__).parent / "frontend"
app = modal.App("medical-chatbot-spa")
chat_histories = modal.Dict.from_name(
    "example-chatbot-spa-history", create_if_missing=True
)


def load_tokenizer_and_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-large",
        device_map="auto",
    )
    return tokenizer, model


gpu_image = (
    modal.Image.debian_slim()
    .pip_install("torch", find_links="https://download.pytorch.org/whl/cu116")
    .pip_install("transformers~=4.31", "accelerate")
    .run_function(load_tokenizer_and_model)
)


with gpu_image.imports():
    import torch

    tokenizer, model = load_tokenizer_and_model()


@app.function(
    mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")]
)
@modal.asgi_app()
def transformer():
    app = fastapi.FastAPI()

    @app.post("/chat")
    def chat(body: dict = fastapi.Body(...)):
        message = body["message"]
        chat_id = body.get("id")
        id, response = generate_response.remote(message, chat_id)
        return JSONResponse({"id": id, "response": response})

    app.mount("/", StaticFiles(directory="/assets", html=True))
    return app


@app.function(gpu="any", image=gpu_image)
def generate_response(
    message: str, id: Optional[str] = None, max_history_length: int = 100
) -> Tuple[str, str]:
    # new_input_ids = tokenizer.encode(
    #     message + tokenizer.eos_token, return_tensors="pt"
    # ).to("cuda")
    new_input_ids = tokenizer(message+tokenizer.eos_token , return_)
    if id is not None and id in chat_histories:
        # Retrieve the existing chat history
        chat_history = chat_histories[id]
        # Concatenate the new input with the previous chat history
        bot_input_ids = torch.cat([chat_history, new_input_ids], dim=-1)
    else:
        # If no id or chat history exists, start a new one
        id = str(uuid.uuid4())
        bot_input_ids = new_input_ids

    # Generate a response
    chat_history = model.generate(
        bot_input_ids,
        max_length=1250,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True       )

    response = tokenizer.decode(
        chat_history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True
    )

    # Trim the chat history if it exceeds the max length
    if chat_history.size(1) > max_history_length:
        chat_history = chat_history[:, -max_history_length:]

    chat_histories[id] = chat_history

    return id, response



@app.local_entrypoint()
def test_response(message: str):
    _, response = generate_response.remote(message)
    print(response)