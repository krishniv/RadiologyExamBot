import os
import json
import random
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hashlib
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure you have the necessary NLTK data




# Paths
images_folder = '/Users/krishnaniveditha/Desktop/LLMprojects/RadiologyQuizBot/src/images'
captions_file = '/Users/krishnaniveditha/Desktop/LLMprojects/roco-dataset/data/validation/radiology/captions.txt'
output_json = './rad_quiz.json'
descriptions_only = "/Users/krishnaniveditha/Desktop/LLMprojects/RadiologyExamBot/src/descriptions.txt"

# Load descriptions and keywords
def load_data(captions_file):
    descriptions = {}
    keywords = {}

    with open(captions_file, 'r') as desc_file:
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


        
def clean_text(text):
    stemmer = PorterStemmer()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Stem the tokens
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(stemmed_tokens)

def hash_text(text):
    # Hash the cleaned text to detect duplicates
    return hashlib.md5(text.encode()).hexdigest()

def read_description(captions_file, output_file):
    unique_hashes = set()
    unique_descriptions = {}

    with open(captions_file, 'r') as desc_file:
        for line in desc_file:
            image, desc = line.strip().split('\t', 1)
            
            # Clean the description but keep original case for output
            cleaned_desc = clean_text(desc)
            text_hash = hash_text(cleaned_desc)
            
            # Check for uniqueness based on the hash
            if text_hash not in unique_hashes:
                unique_hashes.add(text_hash)
                unique_descriptions[text_hash] = desc  # Store the original case description
    
    # Write unique descriptions to the output file
    with open(output_file, 'w') as out_file:
        for desc in unique_descriptions.values():
            out_file.write(f'{desc}\n')

# Run the process
if __name__ == "__main__":
    read_description(captions_file,descriptions_only)
    