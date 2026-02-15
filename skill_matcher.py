import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

# Try multiple ways to load the spacy model
def load_spacy_model():
    """Load spacy model with fallback options for deployment environments"""
    model_name = "en_core_web_sm"
    
    # Try loading the model
    try:
        return spacy.load(model_name)
    except OSError:
        pass
    
    # Try with full path (for some deployment environments)
    try:
        import en_core_web_sm
        return en_core_web_sm.load()
    except (ImportError, OSError):
        pass
    
    # If all fails, raise clear error
    raise OSError(
        f"Spacy model '{model_name}' not found. "
        "Make sure requirements.txt includes the spacy model URL."
    )

nlp = load_spacy_model()

SKILL_DATABASE = {
    "programming_languages": {
        "python", "java", "javascript", "c++", "c#", "ruby", "golang", "rust", "typescript",
        "php", "swift", "kotlin", "scala", "matlab", "perl", "shell", "bash",
        "pascal", "objective-c", "groovy", "lua", "haskell", "erlang", "elixir", "clojure"
    },
    "web_technologies": {
        "html", "css", "html5", "css3", "react", "reactjs", "react.js", "angular", "angularjs",
        "vue", "vuejs", "vue.js", "nodejs", "node.js", "express", "expressjs", "django", 
        "flask", "fastapi", "spring boot", "spring", "asp.net", "rest api", "restful api",
        "graphql", "webpack", "sass", "scss", "less", "bootstrap", "tailwind css", "tailwind",
        "jquery", "ajax", "json", "xml"
    },
    "data_science_ml": {
        "machine learning", "deep learning", "tensorflow", "pytorch", "keras", "scikit-learn",
        "sklearn", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly", "opencv",
        "nlp", "natural language processing", "nltk", "spacy", "transformers", "huggingface",
        "bert", "gpt", "llm", "computer vision", "neural networks", "cnn", "rnn", "lstm",
        "reinforcement learning", "xgboost", "lightgbm", "catboost", "random forest",
        "artificial intelligence", "data analysis", "data science", "statistics",
        "data visualization", "feature engineering", "model deployment"
    },
    "data_engineering": {
        "sql", "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
        "apache kafka", "kafka", "apache spark", "spark", "hadoop", "hive", "apache airflow",
        "airflow", "etl", "data pipeline", "data warehousing", "aws", "gcp", "google cloud",
        "azure", "snowflake", "databricks", "dbt", "oracle", "sqlite", "nosql", "bigquery"
    },
    "devops_cloud": {
        "docker", "kubernetes", "k8s", "jenkins", "terraform", "ansible", "puppet", "chef",
        "aws ec2", "ec2", "aws s3", "s3", "aws lambda", "lambda", "cloudformation", "cicd",
        "ci/cd", "devops", "git", "github", "gitlab", "bitbucket", "jira", "circleci",
        "github actions", "travis ci", "azure devops", "aws cloudwatch", "prometheus", "grafana"
    },
    "tools_frameworks": {
        "git", "docker", "kubernetes", "jupyter", "jupyter notebook", "vscode", "visual studio",
        "intellij", "pycharm", "eclipse", "postman", "swagger", "linux", "unix", "ubuntu",
        "windows server", "vim", "npm", "yarn", "pip", "conda", "virtualenv", "poetry",
        "tableau", "power bi", "powerbi", "excel", "jira", "confluence", "slack"
    },
    "soft_skills": {
        "leadership", "communication", "teamwork", "problem solving", "problem-solving",
        "analytical thinking", "project management", "agile methodology", "scrum master",
        "time management", "presentation skills", "critical thinking", "adaptability",
        "collaboration", "team player", "mentoring", "stakeholder management"
    },
    "methodologies": {
        "agile", "scrum", "kanban", "waterfall", "devops", "tdd", "test driven development",
        "bdd", "behavior driven development", "microservices", "oauth", "jwt", "rest",
        "restful", "soap", "api design", "system design", "oop", "object oriented programming"
    },
    "databases": {
        "mysql", "postgresql", "postgres", "mongodb", "oracle database", "oracle",
        "sql server", "microsoft sql server", "redis", "elasticsearch", "cassandra",
        "dynamodb", "firebase", "mariadb", "sqlite", "couchdb", "neo4j", "graph database"
    },
    "frameworks": {
        "react", "reactjs", "angular", "angularjs", "vue", "vuejs", "django", "flask",
        "spring", "spring boot", "laravel", "ruby on rails", "rails", "asp.net", ".net core",
        "nodejs", "express", "expressjs", "fastapi", "nestjs", "nextjs", "next.js", "nuxtjs"
    },
    "testing": {
        "unit testing", "integration testing", "pytest", "junit", "jest", "mocha", "selenium",
        "cypress", "testng", "cucumber", "test automation", "qa", "quality assurance"
    },
    "other_technical": {
        "api", "microservices", "websockets", "grpc", "message queue", "rabbitmq",
        "celery", "multithreading", "async programming", "design patterns", "algorithms",
        "data structures", "version control", "code review", "debugging", "performance optimization"
    }
}

ALL_SKILLS = set()
for category in SKILL_DATABASE:
    ALL_SKILLS.update(SKILL_DATABASE[category])

# Sort skills by length (longest first) for better matching
ALL_SKILLS_SORTED = sorted(ALL_SKILLS, key=len, reverse=True)

# Common English words to exclude
STOPWORDS = {
    "the", "and", "for", "with", "are", "from", "that", "this", "have", "has",
    "was", "were", "been", "being", "will", "would", "could", "should", "may",
    "can", "our", "your", "their", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "only", "own", "same", "than",
    "too", "very", "just", "but", "not", "now", "also", "well", "back", "even",
    "still", "way", "take", "make", "good", "new", "first", "last", "long", "great",
    "little", "use", "find", "give", "tell", "ask", "work", "seem", "feel", "try",
    "leave", "call", "keep", "let", "begin", "help", "show", "hear", "play", "run",
    "move", "like", "live", "believe", "hold", "bring", "happen", "write", "sit",
    "stand", "lose", "pay", "meet", "include", "continue", "set", "learn", "change",
    "lead", "understand", "watch", "follow", "stop", "create", "speak", "read", "allow",
    "add", "spend", "grow", "open", "walk", "win", "offer", "remember", "love", "consider"
}

def extract_skills(text):
    """
    Extract technical skills from text using exact phrase matching.
    Prioritizes longer skill phrases and filters out common words.
    """
    text_lower = text.lower()
    found_skills = set()
    
    # Add word boundaries and punctuation for better matching
    text_padded = f' {text_lower} '
    
    # Match skills in order of length (longest first to avoid partial matches)
    for skill in ALL_SKILLS_SORTED:
        # Skip very short skills that might be common words
        if len(skill) <= 2:
            continue
            
        # Skip if skill is a common stopword
        if skill in STOPWORDS:
            continue
        
        # Create pattern with word boundaries
        # Match skill surrounded by spaces, punctuation, or line boundaries
        pattern = r'(?:^|\s|[,.\-/\(\)\[\]{}:;])' + re.escape(skill) + r'(?:$|\s|[,.\-/\(\)\[\]{}:;])'
        
        if re.search(pattern, text_padded, re.IGNORECASE):
            found_skills.add(skill)
            # Remove matched skill from text to avoid substring matches
            text_padded = re.sub(pattern, ' ', text_padded, flags=re.IGNORECASE)
    
    return list(found_skills)


def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)


def calculate_similarity(resume_text, job_description):
    """
    Calculate similarity between resume and job description.
    Uses both TF-IDF text similarity and skill matching.
    """
    resume_clean = preprocess_text(resume_text)
    jd_clean = preprocess_text(job_description)

    documents = [resume_clean, jd_clean]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(job_description))
    
    if jd_skills:
        skill_match = len(resume_skills & jd_skills) / len(jd_skills)
        # Weight skill matching more heavily (50-50 split)
        similarity = (similarity * 0.5) + (skill_match * 0.5)

    return similarity * 100


def skill_gap_analysis(resume_text, job_description):
    """
    Analyze skill gaps between resume and job description.
    Returns matched, missing, and all skills found.
    """
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    
    resume_set = set(resume_skills)
    job_set = set(job_skills)

    missing_skills = job_set - resume_set
    matched_skills = resume_set & job_set

    return {
        "matched": sorted(list(matched_skills)),
        "missing": sorted(list(missing_skills)),
        "resume_skills": sorted(list(resume_set)),
        "job_skills": sorted(list(job_set))
    }


def ats_check(resume_text, job_description):
    """
    Check ATS (Applicant Tracking System) compatibility.
    Evaluates resume structure, formatting, and keyword optimization.
    """
    score = 100
    issues = []
    
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()
    
    # Contact information check
    has_email = bool(re.search(r'[\w.-]+@[\w.-]+\.\w+', resume_text))
    has_phone = bool(re.search(r'[\d\s\-\(\)]{10,}', resume_text))
    has_name = len(re.findall(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', resume_text)) > 0
    
    if not has_email:
        score -= 10
        issues.append("Missing email address")
    if not has_phone:
        score -= 10
        issues.append("Missing phone number")
    if not has_name:
        score -= 5
        issues.append("Name not clearly identified")
    
    # Section presence check
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
            score -= 8
    
    if missing_sections:
        issues.append(f"Missing sections: {', '.join(missing_sections)}")
    
    # Word count check
    resume_words = len(resume_text.split())
    if resume_words < 150:
        score -= 15
        issues.append("Resume too short (add more details)")
    elif resume_words > 1200:
        score -= 5
        issues.append("Resume too long (consider condensing)")
    
    # Skill matching with improved extraction
    jd_keywords = extract_skills(jd_lower)
    resume_keywords = extract_skills(resume_lower)
    matched_kw = set(jd_keywords) & set(resume_keywords)
    keyword_match_rate = len(matched_kw) / len(jd_keywords) * 100 if jd_keywords else 0
    
    if keyword_match_rate < 30:
        score -= 20
        issues.append(f"Very low keyword match ({keyword_match_rate:.0f}%) - add relevant skills")
    elif keyword_match_rate < 50:
        score -= 10
        issues.append(f"Low keyword match ({keyword_match_rate:.0f}%) - improve skill alignment")
    elif keyword_match_rate >= 70:
        score += 5
    
    # Formatting check
    bullet_count = resume_text.count('•') + resume_text.count('-') + resume_text.count('*')
    if bullet_count < 5:
        score -= 5
        issues.append("Use more bullet points for better readability")
    
    # Education check
    education_keywords = ["bachelor", "master", "phd", "degree", "university", "college", "diploma", "certification"]
    has_education = any(kw in resume_lower for kw in education_keywords)
    if not has_education:
        score -= 5
        issues.append("Education/certification section not clearly identified")
    
    score = max(0, min(100, score))
    
    return {
        "score": score,
        "issues": issues,
        "keyword_match_rate": keyword_match_rate,
        "word_count": resume_words,
        "has_contact_info": has_email and has_phone,
        "section_count": 3 - len(missing_sections)
    }
