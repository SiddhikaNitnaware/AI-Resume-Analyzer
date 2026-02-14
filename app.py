import streamlit as st
from resume_parser import extract_text_from_pdf
from skill_matcher import calculate_similarity, skill_gap_analysis
import tempfile

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI-Powered Resume Screening System")

st.write("Upload your resume and compare it with a job description.")

# Upload Resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Paste Job Description
job_description = st.text_area("Paste Job Description Here")

if uploaded_file is not None and job_description:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        resume_path = tmp.name

    # Extract Resume Text
    resume_text = extract_text_from_pdf(resume_path)

    # Calculate Score
    score = calculate_similarity(resume_text, job_description.lower())

    # Skill Gap Analysis
    resume_skills, missing_skills = skill_gap_analysis(resume_text, job_description.lower())

    st.subheader("📊 Match Score")
    st.success(f"{score:.2f}% Match")

    st.subheader("✅ Skills Found")
    st.write(", ".join(resume_skills) if resume_skills else "No matching skills detected.")

    st.subheader("❌ Missing Skills (Recommended)")
    st.write(", ".join(missing_skills) if missing_skills else "No major skill gaps!")