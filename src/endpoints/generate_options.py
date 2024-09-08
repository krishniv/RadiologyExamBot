from fastapi import APIRouter, Query
import os
import random
import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("API_KEY")

router = APIRouter()

# Paths
images_folder = '/Users/krishnaniveditha/Desktop/LLMprojects/roco-dataset/data/validation/radiology/images'
descriptions_txt = '/Users/krishnaniveditha/Desktop/LLMprojects/RadiologyExamBot/src/descriptions.txt'

# Load corpus once
with open(descriptions_txt, 'r') as file:
    corpus = [line.strip() for line in file]

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)

state = {}

def get_image_description(image_path):
    """Generate a description for the image using OpenAI."""
    with open(image_path, "rb") as image_file:
        image = Image.open(image_file)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # prompt = f"Generate a concise medical description for the following image: \n\n (Insert description based on this: Axial computed tomography scan of the pelvis showing a diffuse infiltration of the bladder wall, catheter in situ)"

    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",  # Use the appropriate model
    #     messages=[
    #         {"role": "system", "content": "You are a helpful medical assistant."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     max_tokens=150,
    #     temperature=0.5
    # )

    # description = response['choices'][0]['message']['content'].strip()
    description = "Axial computed tomography scan of the pelvis showing a diffuse infiltration of the bladder wall, catheter in situ (arrow)."
    return description


def load_image(folder_path):
    """Load a random image from the folder."""
    if 'image_files' not in state or 'selected_images' not in state:
        state['image_files'] = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        state['selected_images'] = state['image_files'][:]
        random.shuffle(state['selected_images'])

    if not state['selected_images']:
        state['selected_images'] = state['image_files'][:]
        random.shuffle(state['selected_images'])

    selected_image = state['selected_images'].pop()
    return os.path.join(folder_path, selected_image),selected_image

def get_similar_options(target_desc, corpus, tfidf_matrix, vectorizer):
    target_vec = vectorizer.transform([target_desc])
    similarities = cosine_similarity(target_vec, tfidf_matrix).flatten()
    similar_indices = similarities.argsort()[-4:][::-1]
    similar_descs = [corpus[i] for i in similar_indices if corpus[i] != target_desc]
    return similar_descs[:3]

@router.get("/generate/{amount}")
async def generate_options(amount: int):
    """
    Generate a JSON mapping for quiz options for the requested amount of questions.
    The `amount` parameter specifies how many quiz questions to generate.
    """

    quiz_data = []

    for _ in range(amount):
        question_data = {}

        selected_image_path,selected_image= load_image(images_folder)
        correct_desc = get_image_description(selected_image_path)
        similar_options = get_similar_options(correct_desc, corpus, tfidf_matrix, vectorizer)

        all_options = [correct_desc] + similar_options
        random.shuffle(all_options)

        question_data = {
            "image": selected_image,
            "correct": correct_desc,
            "options": all_options
        }

        quiz_data.append(question_data)

    return {"questions": quiz_data}