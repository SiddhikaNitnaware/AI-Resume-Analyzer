from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(resume_text, job_description):
    documents = [resume_text, job_description]

    vectorizer = CountVectorizer().fit_transform(documents)
    similarity_matrix = cosine_similarity(vectorizer)

    return similarity_matrix[0][1] * 100


def extract_skills(text, skill_list):
    found_skills = []

    for skill in skill_list:
        if skill.lower() in text:
            found_skills.append(skill)

    return found_skills


def skill_gap_analysis(resume_text, job_description):
    # Predefined important industry skills (can expand later)
    skill_database = [
        "python", "machine learning", "nlp", "sql", "git",
        "pandas", "scikit-learn", "data analysis",
        "deep learning", "tensorflow", "communication",
        "problem solving", "statistics"
    ]

    resume_skills = extract_skills(resume_text, skill_database)
    jd_skills = extract_skills(job_description, skill_database)

    missing_skills = list(set(jd_skills) - set(resume_skills))

    return resume_skills, missing_skills
