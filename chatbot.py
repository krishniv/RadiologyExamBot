
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
app = modal.App("Medical chabot-spa")
chat_histories = modal.Dict.from_name(
    "example-chatbot-spa-history", create_if_missing=True
)


def load_tokenizer_and_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    # Define BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16)

    # Model name
    model_name = "ruslanmv/Medical-Llama3-v2"

    # Load tokenizer and model with BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, bnb_config=bnb_config)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=bnb_config)



gpu_image = (
    modal.Image.debian_slim()
    .pip_install("torch", find_links="https://download.pytorch.org/whl/cu116")
    .pip_install("transformers~=4.31", "accelerate","bitsandbytes")
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
    message: str, id: Optional[str] = None
) -> Tuple[str, str]:
    sys_message = ''' 
    You are Medical AI Assistant. Please be thorough and provide an informative answer. 
    If you don't know the answer to a specific medical inquiry, advise seeking professional help.
    '''   
    # Create messages structured for the chat template

    if id is not None:
        chat_history = chat_histories[id]
        bot_input_ids = torch.cat([chat_history, message], dim=-1)
    else:
        id = str(uuid.uuid4())
        bot_input_ids = message
    messages = [{"role": "system", "content": sys_message}, {"role": "user", "content": bot_input_ids}]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1000)

    # new_input_ids = tokenizer.encode(
    #     inputs + tokenizer.eos_token, return_tensors="pt"
    # ).to("cuda")


    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1000, use_cache=True)
    # Extract and return the generated text, removing the prompt
    response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    chat_histories[id] = chat_history
    return id, response_text


@app.local_entrypoint()
def test_response(message: str):
    _, response = generate_response.remote(message)
    print(response)