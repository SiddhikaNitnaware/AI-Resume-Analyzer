import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import sys
import subprocess

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

SKILL_DATABASE = {
    "programming_languages": {
        "python", "java", "javascript", "c++", "c#", "ruby", "go", "rust", "typescript",
        "php", "swift", "kotlin", "scala", "r", "matlab", "perl", "shell", "bash"
    },
    "web_technologies": {
        "html", "css", "react", "reactjs", "angular", "vue", "nodejs", "node.js",
        "express", "django", "flask", "fastapi", "spring", "asp.net", "rest api",
        "graphql", "webpack", "sass", "less", "bootstrap", "tailwind"
    },
    "data_science_ml": {
        "machine learning", "deep learning", "tensorflow", "pytorch", "keras", "scikit-learn",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "opencv", "nlp", "nltk",
        "spacy", "transformers", "huggingface", "bert", "gpt", "computer vision",
        "neural networks", "reinforcement learning", "xgboost", "lightgbm", "catboost"
    },
    "data_engineering": {
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "kafka",
        "spark", "hadoop", "hive", "airflow", "etl", "data pipeline", "aws", "gcp",
        "azure", "snowflake", "databricks", "dbt"
    },
    "devops_cloud": {
        "docker", "kubernetes", "jenkins", "terraform", "ansible", "puppet", "chef",
        "aws", "azure", "gcp", "ec2", "s3", "lambda", "cloudformation", "cicd",
        "git", "github", "gitlab", "bitbucket", "jira"
    },
    "tools_frameworks": {
        "git", "docker", "kubernetes", "jupyter", "vscode", "intellij", "eclipse",
        "postman", "swagger", "linux", "unix", "windows server", "vim", "npm",
        "yarn", "pip", "conda", "virtualenv"
    },
    "soft_skills": {
        "leadership", "communication", "teamwork", "problem-solving", "analytical",
        "project management", "agile", "scrum", "time management", "presentation"
    },
    "methodologies": {
        "agile", "scrum", "kanban", "waterfall", "devops", "ci/cd", "tdd", "bdd",
        "microservices", "restful", "oauth", "jwt"
    }
}

ALL_SKILLS = set()
for category in SKILL_DATABASE:
    ALL_SKILLS.update(SKILL_DATABASE[category])

def fuzzy_match(word, threshold=0.8):
    from difflib import SequenceMatcher
    word = word.lower().strip()
    matches = []
    for skill in ALL_SKILLS:
        ratio = SequenceMatcher(None, word, skill).ratio()
        if ratio >= threshold:
            matches.append((skill, ratio))
    if matches:
        return max(matches, key=lambda x: x[1])[0]
    return None

def extract_skills(text):
    text = text.lower()
    doc = nlp(text)
    found_skills = set()
    
    words = re.findall(r'\b\w+\b', text)
    for word in words:
        if word in ALL_SKILLS:
            found_skills.add(word)
    
    for skill in ALL_SKILLS:
        if skill in text:
            found_skills.add(skill)
    
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        for skill in ALL_SKILLS:
            if skill in chunk_text:
                found_skills.add(skill)
    
    return list(found_skills)


def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)


def calculate_similarity(resume_text, job_description):
    resume_clean = preprocess_text(resume_text)
    jd_clean = preprocess_text(job_description)

    documents = [resume_clean, jd_clean]

    vectorizer = CountVectorizer().fit_transform(documents)
    similarity_matrix = cosine_similarity(vectorizer)

    return similarity_matrix[0][1] * 100


def extract_keywords(text):
    text = text.lower()
    return extract_skills(text)


def skill_gap_analysis(resume_text, job_description):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    
    resume_set = set(resume_skills)
    job_set = set(job_skills)

    missing_skills = job_set - resume_set
    matched_skills = resume_set & job_set

    return {
        "matched": list(matched_skills),
        "missing": list(missing_skills),
        "resume_skills": list(resume_set),
        "job_skills": list(job_set)
    }


def ats_check(resume_text, job_description):
    score = 100
    issues = []
    suggestions = []
    
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()
    
    has_email = bool(re.search(r'[\w.-]+@[\w.-]+\.\w+', resume_text))
    has_phone = bool(re.search(r'[\d]{10,}', resume_text))
    has_name = len(re.findall(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', resume_text)) > 0
    
    if not has_email:
        score -= 10
        issues.append("Missing email address")
        suggestions.append("Add a professional email address")
    if not has_phone:
        score -= 10
        issues.append("Missing phone number")
        suggestions.append("Add a contact phone number")
    if not has_name:
        score -= 15
        issues.append("Name not clearly identified")
        suggestions.append("Ensure your full name is at the top of the resume")
    
    sections = {
        "experience": ["experience", "work history", "employment", "professional background"],
        "education": ["education", "academic", "degree", "university", "college"],
        "skills": ["skills", "technical skills", "competencies", "expertise"],
    }
    
    missing_sections = []
    for section, keywords in sections.items():
        found = any(kw in resume_lower for kw in keywords)
        if not found:
            missing_sections.append(section)
            score -= 8
    
    if missing_sections:
        issues.append(f"Missing sections: {', '.join(missing_sections)}")
        suggestions.append(f"Add these sections: {', '.join(missing_sections)}")
    
    resume_words = len(resume_text.split())
    if resume_words < 100:
        score -= 15
        issues.append("Resume too short")
        suggestions.append("Add more details to your resume (aim for 300-500 words)")
    elif resume_words > 1500:
        score -= 10
        issues.append("Resume too long")
        suggestions.append("Consider condensing your resume to 1-2 pages")
    
    jd_keywords = extract_keywords(jd_lower)
    resume_keywords = extract_keywords(resume_lower)
    matched_kw = set(jd_keywords) & set(resume_keywords)
    keyword_match_rate = len(matched_kw) / len(jd_keywords) * 100 if jd_keywords else 0
    
    if keyword_match_rate < 30:
        score -= 20
        issues.append(f"Low keyword match ({keyword_match_rate:.0f}%)")
        suggestions.append("Include more keywords from the job description")
    
    bullet_count = resume_text.count('•') + resume_text.count('-') + resume_text.count('*')
    if bullet_count < 5:
        score -= 5
        issues.append("Limited use of bullet points")
        suggestions.append("Use bullet points for easier scanning")
    
    score = max(0, score)
    
    return {
        "score": score,
        "issues": issues,
        "suggestions": suggestions,
        "keyword_match_rate": keyword_match_rate,
        "word_count": resume_words,
        "has_contact_info": has_email and has_phone,
        "section_count": 3 - len(missing_sections)
    }


def generate_improvement_tips(resume_text, job_description, skill_analysis, ats_results):
    tips = []
    
    if ats_results["keyword_match_rate"] < 50:
        missing_kw = set(extract_keywords(job_description.lower())) - set(extract_keywords(resume_text.lower()))
        if missing_kw:
            top_missing = list(missing_kw)[:5]
            tips.append(f"Add these relevant keywords: {', '.join(top_missing)}")
    
    if skill_analysis["missing"]:
        tips.append(f"Consider highlighting these skills: {', '.join(skill_analysis['missing'][:5])}")
    
    if ats_results["issues"]:
        for suggestion in ats_results["suggestions"][:3]:
            tips.append(suggestion)
    
    if ats_results["word_count"] < 300:
        tips.append("Add more quantifiable achievements (e.g., 'Improved efficiency by 30%')")
    
    resume_lower = resume_text.lower()
    has_action_verbs = any(verb in resume_lower for verb in ['led', 'managed', 'developed', 'implemented', 'created', 'achieved', 'reduced'])
    if not has_action_verbs:
        tips.append("Use action verbs to start bullet points (e.g., Led, Developed, Implemented)")
    
    if not tips:
        tips.append("Your resume looks good! Keep tailoring it for each application.")
    
    return tips
