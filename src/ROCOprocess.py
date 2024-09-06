import os
import json
import random
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Paths
images_folder = '/Users/krishnaniveditha/Desktop/LLMprojects/RadiologyQuizBot/src/images'
descriptions_file = '/Users/krishnaniveditha/Desktop/LLMprojects/roco-dataset/data/validation/radiology/captions.txt'
output_json = './rad_quiz.json'

# Load descriptions and keywords
def load_data(descriptions_file):
    descriptions = {}
    keywords = {}

    with open(descriptions_file, 'r') as desc_file:
        for line in desc_file:
            image, desc = line.strip().split('\t', 1)
            descriptions[image.strip()+ ".jpg"] = desc.strip()

    return descriptions

# Check if an image is present in the folder
def get_valid_images(images_folder, descriptions):
    available_images = set(os.listdir(images_folder))
    valid_images = [img for img in descriptions if img in available_images]
    return valid_images

# Create text options by finding similar descriptions based on keywords
def get_similar_options(target_desc, descriptions):
    corpus = list(descriptions.values())
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Compute similarity for target description
    target_vec = vectorizer.transform([target_desc])
    similarities = cosine_similarity(target_vec, tfidf_matrix).flatten()
    filtered_similarities = np.array(list(set(similarities)))
    # Get the top 3 most similar descriptions (excluding the target itself)
    similar_indices = filtered_similarities.argsort()[-4:][::-1]
    similar_descs = [corpus[i] for i in similar_indices if corpus[i] != target_desc]
    return similar_descs[:3]

# Main function to create the JSON mapping
def create_quiz_mapping(images_folder, descriptions):
    valid_images = get_valid_images(images_folder, descriptions)
    quiz_data = {}

    for image in valid_images:
        correct_desc = descriptions[image]
        # Get three similar options
        similar_options = get_similar_options(correct_desc, descriptions)
        all_options = [correct_desc] + similar_options
        random.shuffle(all_options)

        quiz_data[image] = {
            "correct": correct_desc,
            "options": all_options
        }
    return quiz_data

# Save JSON data
def save_json(data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Run the process
if __name__ == "__main__":
    descriptions = load_data(descriptions_file)
    quiz_data = create_quiz_mapping(images_folder, descriptions)
    save_json(quiz_data, output_json)

    print(f"Quiz JSON created successfully: {output_json}")
