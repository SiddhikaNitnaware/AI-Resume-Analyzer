import streamlit as st
from resume_parser import extract_text_from_pdf
from skill_matcher import (
    calculate_similarity,
    skill_gap_analysis,
    ats_check,
    generate_improvement_tips,
    extract_skills
)
import tempfile

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI-Powered Resume Screening System")
st.write("Upload your resume and compare it with a job description.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
if uploaded_file is not None:
    if uploaded_file.size > 4 * 1024 * 1024:
        st.error("File size must be less than 4MB")
        uploaded_file = None
        
job_description = st.text_area("Paste Job Description Here")

submit = st.button("Analyze Resume", type="primary")

if submit and uploaded_file is not None and job_description:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        resume_path = tmp.name

    resume_text = extract_text_from_pdf(resume_path)

    with st.spinner("Analyzing resume..."):
        score = calculate_similarity(resume_text, job_description.lower())
        skill_analysis = skill_gap_analysis(resume_text, job_description.lower())
        ats_results = ats_check(resume_text, job_description.lower())
        improvement_tips = generate_improvement_tips(resume_text, job_description, skill_analysis, ats_results)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Match Score")
        st.success(f"{score:.2f}% Match")
    
    with col2:
        st.subheader("🤖 ATS Score")
        if ats_results["score"] >= 80:
            st.success(f"{ats_results['score']}/100 - ATS Friendly")
        elif ats_results["score"] >= 60:
            st.warning(f"{ats_results['score']}/100 - Needs Improvement")
        else:
            st.error(f"{ats_results['score']}/100 - Needs Work")

    tab1, tab2, tab3, tab4 = st.tabs(["✅ Skills", "❌ Missing Skills", "🔍 ATS Check", "💡 Tips"])

    with tab1:
        st.write("**Skills Found in Resume:**")
        if skill_analysis["resume_skills"]:
            skills_html = " ".join([f"<span style='background-color:#4CAF50;color:white;padding:4px 8px;border-radius:4px;margin:2px;display:inline-block'>{s}</span>" 
                                   for s in skill_analysis["resume_skills"]])
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.write("No matching skills detected.")

    with tab2:
        st.write("**Missing Skills (Recommended):**")
        if skill_analysis["missing"]:
            missing_html = " ".join([f"<span style='background-color:#f44336;color:white;padding:4px 8px;border-radius:4px;margin:2px;display:inline-block'>{s}</span>" 
                                    for s in skill_analysis["missing"]])
            st.markdown(missing_html, unsafe_allow_html=True)
        else:
            st.write("No major skill gaps!")

    with tab3:
        st.write("**ATS Compatibility Details:**")
        st.metric("Keyword Match Rate", f"{ats_results['keyword_match_rate']:.1f}%")
        st.metric("Word Count", ats_results["word_count"])
        
        st.write("**Issues Found:**")
        if ats_results["issues"]:
            for issue in ats_results["issues"]:
                st.warning(f"• {issue}")
        else:
            st.success("No major issues found!")

    with tab4:
        st.write("**Improvement Tips:**")
        for i, tip in enumerate(improvement_tips, 1):
            st.info(f"{i}. {tip}")

    st.divider()
    st.subheader("📋 Resume Analysis Summary")
    
    with st.expander("See full analysis"):
        st.write(f"**Match Score:** {score:.2f}%")
        st.write(f"**ATS Score:** {ats_results['score']}/100")
        st.write(f"**Keywords Matched:** {len(skill_analysis['matched'])}/{len(skill_analysis['job_skills'])}")
        st.write(f"**Resume Word Count:** {ats_results['word_count']}")
