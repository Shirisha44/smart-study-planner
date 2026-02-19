import os
import datetime
import streamlit as st
import pdfplumber
from crewai import Agent, Task, Crew, LLM

# --- SECURE API KEY SETUP ---
# This tells Streamlit to look for a hidden secret password file
# If it can't find the secret (like when you test locally), it uses a backup.
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    API_KEY = "PASTE_YOUR_ACTUAL_API_KEY_HERE_FOR_LOCAL_TESTING"

os.environ["GEMINI_API_KEY"] = API_KEY

# Set up the page with a wide layout and a cool icon
st.set_page_config(page_title="Agentic Study Planner", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# 2. THE LLM
current_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=API_KEY,
    temperature=0.1 
)

# --- HELPER FUNCTION TO READ FILES ---
def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.pdf'):
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
        elif uploaded_file.name.endswith('.txt'):
            text = uploaded_file.getvalue().decode("utf-8")
    return text
# ------------------------------------------

# 3. THE AGENT
study_expert = Agent(
    role='Senior Academic Counselor',
    goal='Create a personalized, time-managed study roadmap for {subject} from {current_date} to {deadline}.',
    backstory='You are an expert in curriculum design. You break down official syllabuses into manageable daily study tasks, estimating realistic study hours and finding the best online resources.',
    llm=current_llm, 
    verbose=True,
    allow_delegation=False,
    max_iter=2
)

# 4. THE TASK
study_task = Task(
    description='''Today is {current_date}. The deadline is {deadline}. You have exactly {days_left} days.
    Subject: {subject}
    
    Syllabus Context:
    {syllabus_text}
    
    Task: Create a {days_left}-day daily study schedule for "{subject}". 
    - Map the topics from the syllabus (if provided) across the {days_left} days.
    - Estimate how many hours should be spent on that day's topic based on its complexity (e.g., 1.5 hours, 3 hours).
    - Include 1 practice question for the day.
    - Provide 1 clickable markdown link to a relevant resource (Use standard sites like Wikipedia, GeeksforGeeks, or generate a YouTube Search link formatted like: [Watch Video](https://www.youtube.com/results?search_query=your+topic+here)).
    - ONLY output the Markdown table. Do not include introductory text.''',
    expected_output='A Markdown table with columns: Day, Date, Topic, Estimated Hours, Practice Question, Resource Link.',
    agent=study_expert
)

# ==========================================
# 🎨 STREAMLIT UI DESIGN (UPGRADED)
# ==========================================

# Main Header
st.title("🎓 Agentic AI Smart Study Planner")
st.markdown("Welcome to your personal AI Academic Counselor. Upload your syllabus, set your deadline, and let the AI generate a day-by-day roadmap complete with study hours and resource links.")
st.divider()

# Sidebar for Inputs
with st.sidebar:
    st.header("⚙️ Plan Configuration")
    
    with st.form("study_form"):
        sub = st.text_input("📚 Subject Name", placeholder="e.g., Machine Learning...")
        date = st.date_input("🗓️ Exam / Deadline Date")
        
        st.markdown("### 📄 Optional Context")
        uploaded_file = st.file_uploader("Upload Syllabus (PDF/TXT)", type=['pdf', 'txt'], help="Upload your curriculum to map topics exactly.")
        
        # Make the button wide and prominent
        submit = st.form_submit_button("🚀 Generate My Roadmap", use_container_width=True)

# Default Empty State screen
if not submit:
    st.info("👈 Please configure your study plan in the sidebar to get started!")
    col1, col2, col3 = st.columns(3)
    col1.metric("Step 1", "Set Subject & Date")
    col2.metric("Step 2", "Upload Syllabus")
    col3.metric("Step 3", "Get Daily Plan")

# Execution State
if submit:
    today = datetime.date.today()
    days_remaining = (date - today).days
    
    if days_remaining <= 0:
        st.sidebar.error("❌ Please select a deadline in the future!")
    else:
        # File processing
        syllabus_content = extract_text_from_file(uploaded_file)
        if not syllabus_content.strip():
            syllabus_content = "No syllabus provided. Please generate a standard comprehensive curriculum based on the subject name."
        else:
            syllabus_content = syllabus_content[:15000] 
            st.sidebar.success("📄 Syllabus successfully loaded!")

        # Main processing area
        with st.spinner(f"🤖 AI is analyzing the curriculum and calculating a {days_remaining}-day schedule..."):
            try:
                crew = Crew(
                    agents=[study_expert],
                    tasks=[study_task],
                    verbose=True
                )
                
                result = crew.kickoff(inputs={
                    'subject': sub, 
                    'deadline': str(date),
                    'current_date': str(today),
                    'days_left': str(days_remaining),
                    'syllabus_text': syllabus_content
                })
                
                # --- UI: DISPLAY RESULTS ---
                st.success("🎉 Your Personalized Study Roadmap is Ready!")
                
                # Show top-level metrics
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Total Days", value=days_remaining)
                col2.metric(label="Subject", value=sub)
                col3.metric(label="Status", value="Optimized 🚀")
                
                st.markdown("### 📅 Daily Schedule")
                
                # Create a nice container for the table
                with st.container(border=True):
                    st.markdown(result.raw)
                
                # NEW: Download Button
                st.download_button(
                    label="📥 Download Plan (Markdown)",
                    data=result.raw,
                    file_name=f"{sub.replace(' ', '_')}_study_plan.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error during generation: {e}")
