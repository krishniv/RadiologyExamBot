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
from endpoints.modelcaption import generate_medical_description
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
    description = generate_medical_description(image_path)
    # description = "Axial computed tomography scan of the pelvis showing a diffuse infiltration of the bladder wall, catheter in situ (arrow)."
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