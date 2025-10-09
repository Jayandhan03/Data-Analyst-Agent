import streamlit as st
import tempfile
import time
from pathlib import Path
from Report_agent import Report_agent
import pandas as pd

st.set_page_config(
    page_title="AI Business Report Generator",
    page_icon="📊",
    layout="wide",
)

# --- Custom CSS for Clean Professional Look ---
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4F46E5;
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #4338CA;
            transform: scale(1.03);
        }
        .report-box {
            background-color: #f8f9fa;
            color: #1e1e2f;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 0 25px rgba(79, 70, 229, 0.2);
            font-size: 1.05rem;
            line-height: 1.8;
            white-space: pre-wrap;
            overflow-x: auto;
            max-height: 700px;
        }
        .report-title {
            color: #4F46E5;
            font-weight: 700;
            font-size: 1.4rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.title("📈 AI Business Report Generator")
st.markdown("#### Upload your dataset and let AI perform in-depth EDA & generate an actionable business report.")
st.markdown("---")

# --- File Upload Section ---
uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx"], help="Upload CSV or Excel file")

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Save to temp CSV for compatibility
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        tmp_path = tmp_file.name

    # --- Progress Simulation ---
    st.markdown("### ⚙️ Generating Business Report...")
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for percent in range(0, 100, 10):
        progress_text.text(f"Analyzing dataset... {percent + 10}%")
        time.sleep(0.2)
        progress_bar.progress(percent + 10)

    # --- Agent Execution ---
    try:
        with st.spinner("🧠 AI is analyzing your data... Please wait."):
            result = Report_agent(tmp_path)

        st.success("✅ Business Report Generated Successfully!")
        st.markdown("---")

        # --- Display Clean Result ---
        st.markdown("<div class='report-title'>📊 Business Analysis Report</div>", unsafe_allow_html=True)

        output_text = result["output"] if isinstance(result, dict) and "output" in result else str(result)
        st.markdown(f"<div class='report-box'>{output_text}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")

else:
    st.info("👆 Upload a CSV or Excel file to begin analysis.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with ❤️ using LangChain, Streamlit, and OpenAI</p>",
    unsafe_allow_html=True
)
