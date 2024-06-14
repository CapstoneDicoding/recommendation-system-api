from flask import Flask, request
from googletrans import Translator
from langdetect import detect
import tensorflow as tf
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

def translate_text(text):
    translator = Translator()
    try:
        # Detect the language
        lang = detect(text)
        # Translate only if the text is in Indonesian
        if lang == 'id':
            translated = translator.translate(text, src='id', dest='en')
            return translated.text
        else:
            # Return the original text if it's not in Indonesian
            return text
    except Exception as e:
        # If detection or translation fails, return the original text
        return text

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords and lemmatize, excluding words containing 'unknown'
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words and 'unknown' not in word)
    return text

def combine_candidate_data(candidate):
    combined_text = []
    # Combine basic info
    basic_info = candidate['basic_info']
    if 'location' in basic_info:
        combined_text.append(translate_text(basic_info['location']))

    # Combine work experience
    for work in candidate['work_experience']:
        combined_text.append(translate_text(work['job_title']))
        if 'job_desc' in work:
            combined_text.extend(translate_text(desc) for desc in work['job_desc'])

    # Combine education
    for education in candidate['education']:
        combined_text.append(translate_text(education['title']))
        if 'description' in education:
            combined_text.extend(translate_text(desc) for desc in education['description'])

    # Combine languages, skills, and certifications
    combined_text.extend(translate_text(language) for language in candidate['languages'])
    combined_text.extend(translate_text(skill) for skill in candidate['skills'])
    for certification in candidate['certification']:
        combined_text.append(translate_text(certification['title']))
        if 'issuer' in certification:
            combined_text.append(translate_text(certification['issuer']))

    # Join all text into a single string
    combined_text = ' '.join(combined_text)
    # Preprocess the combined text
    return preprocess_text(combined_text)

# Function to modify candidates JSON to include CV text
def modify_candidates_data(candidates):
    modified_candidates = []
    for candidate in candidates:
        modified_candidate = {
            'cv_id': candidate['cv_id'],
            'cv_text': combine_candidate_data(candidate)
        }
        modified_candidates.append(modified_candidate)
    return modified_candidates

model = SentenceTransformer('bert-base-uncased')
def calculate_similarity(jobreq, cv_text):
    # Encode job requirement and CV text to get embeddings
    embeddings_jobreq = model.encode(jobreq)
    embeddings_cv = model.encode(cv_text)

    # Normalize embeddings
    embeddings_jobreq_norm = tf.nn.l2_normalize(embeddings_jobreq, axis=0)
    embeddings_cv_norm = tf.nn.l2_normalize(embeddings_cv, axis=0)

    # Compute cosine similarity
    similarity_score = tf.tensordot(embeddings_jobreq_norm, embeddings_cv_norm, axes=1).numpy()

    return similarity_score

def sort_candidates_by_similarity(candidate_output):
    """
    Sorts candidates by their similarity score in descending order.

    Parameters:
    candidate_output (list of dict): List of dictionaries containing 'id' and 'similarity' keys for each candidate.

    Returns:
    list of dict: Sorted list of dictionaries in descending order based on similarity score.
    """
    sorted_candidates = sorted(candidate_output, key=lambda x: x['similarity'], reverse=True)
    return sorted_candidates

@app.route('/', methods=['POST'])
def main():
    request_json = request.get_json(silent=True)
    if not request_json:
        return 'Invalid request', 400
    if 'job_requirements' not in request_json or 'cvs' not in request_json:
        return 'Invalid request', 400

    try:
        jobreq = request_json['job_requirements']
        candidates = request_json['cvs']
        preprocess_jobreq = translate_text(preprocess_text(jobreq))
        modified_candidates = modify_candidates_data(candidates)
        candidate_output = []
        for candidate in modified_candidates:
            cv_id = candidate["cv_id"]
            cv_text = candidate["cv_text"]
            similarity_score = calculate_similarity(preprocess_jobreq, cv_text)
            candidate_output.append({
                "cv_id": cv_id,
                "similarity": float(similarity_score) * 100
            })
        sorted_candidates = sort_candidates_by_similarity(candidate_output)
        return sorted_candidates
    except Exception as e:
        err = f"Error processing request: {e}"
        return err, 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)