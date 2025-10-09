import streamlit as st
import pandas as pd
import os
import time
from langgraph.graph import START, StateGraph
from Cleaner_Agent import DataAnalystAgent, AgentStateModel
from Report_agent import Report_agent # Assuming your reporter code is saved in Reporter_Agent.py

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Integrated Data Analysis & Reporting AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Impressive UI ---
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 3rem;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 3rem;
    }
    h1 {
        color: #1e3d59;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #ff6e40;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff9e80;
    }
    .stFileUploader {
        border: 2px dashed #1e3d59;
        border-radius: 8px;
        padding: 20px;
        background-color: #ffffff;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #1e3d59;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("🚀 AI-Powered Data Pipeline: Clean, Analyze, Report")
    st.markdown("Upload your data, provide instructions, and let the AI agents do the heavy lifting. From raw data to a full business report in minutes.")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        uploaded_file = st.file_uploader("1. Upload your CSV or Excel file", type=["csv", "xlsx"])
        
        instructions = st.text_area(
            "2. Provide instructions for the AI",
            height=150,
            placeholder="e.g., 'Handle all missing values in the sales column by filling with the mean. Remove any duplicate rows. My target variable is customer satisfaction.'"
        )
        
        start_button = st.button("✨ Start Analysis & Reporting")

    # --- Main Content Area ---
    col1, col2 = st.columns((2, 1))

    with col1:
        st.header("📋 Pipeline Status & Final Report")
        status_container = st.container(border=True, height=600)

    with col2:
        st.header("📊 Data Preview")
        preview_container = st.container(border=True)

        if uploaded_file is not None:
            try:
                # Read and display a preview of the uploaded data
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                preview_container.dataframe(df.head())
            except Exception as e:
                preview_container.error(f"Error reading file: {e}")

    if start_button:
        if uploaded_file is None or not instructions:
            st.warning("Please upload a file and provide instructions before starting.")
            return

        # --- Pipeline Execution ---
        try:
            # 1. Save the uploaded file to a temporary path
            temp_dir = "temp_data"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            status_container.info(f"✅ File saved temporarily to: {file_path}")

            # ----------------------------------------------------
            # --- STAGE 1: DATA CLEANING PIPELINE ---
            # ----------------------------------------------------
            status_container.info("🚀 Initiating Data Cleaning Pipeline...")
            time.sleep(1)

            # Initialize the agent and graph
            agent = DataAnalystAgent()
            graph = StateGraph(AgentStateModel)
            graph.add_node("supervisor", agent.supervisor_node)
            graph.add_node("PreprocessingPlanner_node", agent.PreprocessingPlanner_node)
            graph.add_node("Cleaner_node", agent.Cleaner_node)
            graph.add_edge(START, "supervisor")
            
            app = graph.compile()

            # Define the initial state for the cleaning workflow
            initial_state = AgentStateModel(
                Instructions=instructions,
                Path=file_path,
                messages=[],
                Analysis=[],
                next="",
                current_reasoning=""
            )

            # Invoke the cleaning workflow
            with st.spinner("Cleaning agent is thinking..."):
                final_cleaning_state = app.invoke(initial_state)

            # Check for errors in the cleaning phase
            if "error" in str(final_cleaning_state).lower():
                status_container.error("❗️ An error occurred during the cleaning phase. Aborting.")
                status_container.json(final_cleaning_state)
                return
            
            status_container.success("✅ Data Cleaning Pipeline Completed Successfully!")
            st.balloons()

            # ----------------------------------------------------
            # --- STAGE 2: REPORTING PIPELINE ---
            # ----------------------------------------------------
            status_container.info("🚀 Initiating Reporting Pipeline...")
            time.sleep(1)

            # The file at 'file_path' is now cleaned, so we can pass it directly.
            with st.spinner("Report generation in progress... This may take a moment."):
                report_result = Report_agent(df_path=file_path)

            # Display the final report
            final_report = report_result.get("output", "Could not extract the final report.")
            status_container.markdown("---")
            status_container.subheader("📈 Final Business Report")
            status_container.markdown(final_report)
            
            # Clean up the temporary file
            os.remove(file_path)
            status_container.info(f"Cleaned up temporary file: {file_path}")

        except Exception as e:
            st.error(f"An unexpected error occurred during the pipeline execution: {e}")
            # Also display traceback for easier debugging if needed
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()