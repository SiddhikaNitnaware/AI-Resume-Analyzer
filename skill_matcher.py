import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
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
        "php", "swift", "kotlin", "scala", "r", "matlab", "perl", "shell", "bash", "c",
        "pascal", "objective-c", "groovy", "lua", "haskell", "erlang", "elixir", "clojure"
    },
    "web_technologies": {
        "html", "css", "react", "reactjs", "angular", "vue", "vuejs", "nodejs", "node.js",
        "express", "django", "flask", "fastapi", "spring", "asp.net", "rest api", "restful",
        "graphql", "webpack", "sass", "less", "bootstrap", "tailwind", "jquery", "ajax"
    },
    "data_science_ml": {
        "machine learning", "deep learning", "tensorflow", "pytorch", "keras", "scikit-learn",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "opencv", "nlp", "nltk",
        "spacy", "transformers", "huggingface", "bert", "gpt", "computer vision",
        "neural networks", "reinforcement learning", "xgboost", "lightgbm", "catboost",
        "artificial intelligence", "ai", "ml", "data analysis", "data science", "statistics"
    },
    "data_engineering": {
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "kafka",
        "spark", "hadoop", "hive", "airflow", "etl", "data pipeline", "aws", "gcp",
        "azure", "snowflake", "databricks", "dbt", "oracle", "sqlite", "nosql"
    },
    "devops_cloud": {
        "docker", "kubernetes", "jenkins", "terraform", "ansible", "puppet", "chef",
        "ec2", "s3", "lambda", "cloudformation", "cicd", "ci/cd", "devops",
        "git", "github", "gitlab", "bitbucket", "jira", "jenkins", "circleci"
    },
    "tools_frameworks": {
        "git", "docker", "kubernetes", "jupyter", "vscode", "intellij", "eclipse",
        "postman", "swagger", "linux", "unix", "windows server", "vim", "npm",
        "yarn", "pip", "conda", "virtualenv", "jupyter notebook", "tableau", "powerbi"
    },
    "soft_skills": {
        "leadership", "communication", "teamwork", "problem-solving", "analytical thinking",
        "project management", "agile", "scrum", "time management", "presentation",
        "critical thinking", "adaptability", "collaboration", "team player"
    },
    "methodologies": {
        "agile", "scrum", "kanban", "waterfall", "devops", "tdd", "bdd",
        "microservices", "oauth", "jwt", "rest", "api"
    },
    "databases": {
        "mysql", "postgresql", "mongodb", "oracle", "sql server", "redis", "elasticsearch",
        "cassandra", "dynamodb", "firebase", "mariadb", "sqlite"
    },
    "frameworks": {
        "react", "angular", "vue", "django", "flask", "spring", "laravel", "rails",
        ".net", "node", "express", "fastapi", "nestjs", "nextjs", "nuxtjs"
    }
}

ALL_SKILLS = set()
for category in SKILL_DATABASE:
    ALL_SKILLS.update(SKILL_DATABASE[category])

def extract_skills(text):
    text_lower = text.lower()
    found_skills = set()
    
    words = re.findall(r'\b\w+\b', text_lower)
    for word in words:
        if word in ALL_SKILLS:
            found_skills.add(word)
    
    for skill in ALL_SKILLS:
        if skill in text_lower:
            found_skills.add(skill)
    
    doc = nlp(text_lower)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        for skill in ALL_SKILLS:
            if skill in chunk_text:
                found_skills.add(skill)
    
    return list(found_skills)


def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)


def calculate_similarity(resume_text, job_description):
    resume_clean = preprocess_text(resume_text)
    jd_clean = preprocess_text(job_description)

    documents = [resume_clean, jd_clean]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    resume_skills = set(extract_skills(resume_text.lower()))
    jd_skills = set(extract_skills(job_description.lower()))
    
    if jd_skills:
        skill_match = len(resume_skills & jd_skills) / len(jd_skills)
        similarity = (similarity * 0.6) + (skill_match * 0.4)

    return similarity * 100


def skill_gap_analysis(resume_text, job_description):
    resume_skills = extract_skills(resume_text.lower())
    job_skills = extract_skills(job_description.lower())
    
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
    
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()
    
    has_email = bool(re.search(r'[\w.-]+@[\w.-]+\.\w+', resume_text))
    has_phone = bool(re.search(r'[\d]{10,}', resume_text))
    has_name = len(re.findall(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', resume_text)) > 0
    
    if not has_email:
        score -= 8
        issues.append("Missing email address")
    if not has_phone:
        score -= 8
        issues.append("Missing phone number")
    if not has_name:
        score -= 10
        issues.append("Name not clearly identified")
    
    sections = {
        "experience": ["experience", "work history", "employment", "professional background", "work experience"],
        "education": ["education", "academic", "degree", "university", "college", "qualification"],
        "skills": ["skills", "technical skills", "competencies", "expertise", "technologies"],
    }
    
    missing_sections = []
    for section, keywords in sections.items():
        found = any(kw in resume_lower for kw in keywords)
        if not found:
            missing_sections.append(section)
            score -= 6
    
    if missing_sections:
        issues.append(f"Missing sections: {', '.join(missing_sections)}")
    
    resume_words = len(resume_text.split())
    if resume_words < 100:
        score -= 12
        issues.append("Resume too short")
    elif resume_words > 1500:
        score -= 8
        issues.append("Resume too long")
    
    jd_keywords = extract_skills(jd_lower)
    resume_keywords = extract_skills(resume_lower)
    matched_kw = set(jd_keywords) & set(resume_keywords)
    keyword_match_rate = len(matched_kw) / len(jd_keywords) * 100 if jd_keywords else 0
    
    if keyword_match_rate < 40:
        score -= 15
        issues.append(f"Low keyword match ({keyword_match_rate:.0f}%)")
    elif keyword_match_rate > 70:
        score += 5
    
    bullet_count = resume_text.count('•') + resume_text.count('-') + resume_text.count('*')
    if bullet_count < 3:
        score -= 5
        issues.append("Limited use of bullet points")
    
    education_keywords = ["bachelor", "master", "phd", "degree", "university", "college", "diploma"]
    has_education = any(kw in resume_lower for kw in education_keywords)
    if not has_education:
        score -= 8
        issues.append("Education section not found")
    
    score = max(0, min(100, score))
    
    return {
        "score": score,
        "issues": issues,
        "keyword_match_rate": keyword_match_rate,
        "word_count": resume_words,
        "has_contact_info": has_email and has_phone,
        "section_count": 3 - len(missing_sections)
    }
