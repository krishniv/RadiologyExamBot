import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .modelcaption import generate_medical_description

class QuizGenerator:
    def __init__(self):
        self.images_folder = './medical_images'
        self.images_url_base = '/'  # Base URL for frontend access
        self.descriptions_txt = './descriptions.txt'
        self.state = {
            'image_files': [],
            'selected_images': []
        }
        
        # Load corpus and initialize TF-IDF once
        with open(self.descriptions_txt, 'r') as file:
            self.corpus = [line.strip() for line in file]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def load_image(self):
        """Load a random image from the folder."""
        if not self.state['image_files']:
            self.state['image_files'] = [
                f for f in os.listdir(self.images_folder) 
                if os.path.isfile(os.path.join(self.images_folder, f))
            ]
            self.state['selected_images'] = self.state['image_files'][:]
            
        if not self.state['selected_images']:
            self.state['selected_images'] = self.state['image_files'][:]
            random.shuffle(self.state['selected_images'])

        selected_image = self.state['selected_images'].pop()
        image_path = os.path.join(self.images_folder, selected_image)
        return image_path, selected_image

    def get_similar_options(self, target_desc, num_options=3):
        """Get similar descriptions using TF-IDF and cosine similarity."""
        target_vec = self.vectorizer.transform([target_desc])
        similarities = cosine_similarity(target_vec, self.tfidf_matrix).flatten()
        similar_indices = similarities.argsort()[-(num_options + 1):][::-1]
        similar_descs = [self.corpus[i] for i in similar_indices if self.corpus[i] != target_desc]
        return similar_descs[:num_options]

    def generate_question(self):
        """Generate a single question with options."""
        image_path, image_name = self.load_image()
        image_url = f"{self.images_url_base}{image_name}"  # Create URL for frontend
        correct_desc = generate_medical_description(image_path)
        similar_options = self.get_similar_options(correct_desc)
        
        all_options = [correct_desc] + similar_options
        random.shuffle(all_options)
        
        return {
            "image": image_url,  # Return URL instead of just filename
            "correct": correct_desc,
            "options": all_options
        }

# Initialize quiz generator once
quiz_generator = QuizGenerator()