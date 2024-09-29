from transformers import AutoProcessor, AutoModelForVisionTextDualEncoder
from PIL import Image
import torch
import requests
from io import BytesIO

# Load the processor and model
model_name = "wisdomik/Quilt-Llava-v1.5-7b"  # Ensure this model supports multimodal tasks
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVisionTextDualEncoder.from_pretrained(model_name)

# Load the image
image_url = "https://wisdomikezogwo.github.io/images/eval_example_3_.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Text input
question = "What do you see in this image?"

# Prepare inputs for the model (image and text)
inputs = processor(text=question, images=image, return_tensors="pt", padding=True)

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate prediction
with torch.no_grad():
    outputs = model(**inputs)

# Decode the outputs (e.g., logits or answer)
print(outputs)  # This will provide the logits or the generated response
