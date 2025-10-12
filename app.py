import streamlit as st
import pandas as pd
import os
import time
import re
import traceback
import shutil
from langgraph.graph import START, StateGraph, END

# --- Import Agent Logic ---
# Note: Ensure these files are in the same directory or your Python path
from Cleaner_Agent import DataAnalystAgent, AgentStateModel
from Report_agent import Report_agent
from Visualizer_agent import Visualizer_agent

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Data Science Pipeline",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Professional & Impressive UI ---
st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #F0F2F6;
    }

    /* Page Title */
    h1 {
        color: #1E3D59;
        font-family: 'Georgia', serif;
        font-weight: bold;
        text-align: center;
    }
    
    /* Markdown Description */
    .st-emotion-cache-16txtl3 > p {
        text-align: center;
        color: #4A4A4A;
        font-size: 1.1rem;
    }

    /* Sidebar Styling */
    .st-sidebar {
        background-color: #FFFFFF;
        border-right: 2px solid #E0E0E0;
    }
    .st-sidebar h2 {
        color: #1E3D59;
    }

    /* Main Content Headers */
    .st-emotion-cache-10trblm h2 {
        color: #1E3D59;
        border-bottom: 2px solid #FF6E40;
        padding-bottom: 10px;
    }

    /* Start Button */
    .stButton>button {
        background-image: linear-gradient(to right, #FF6E40 0%, #FF9E80 51%, #FF6E40 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        transition: 0.5s;
        background-size: 200% auto;
        box-shadow: 0 0 20px #eee;
        width: 100%;
    }
    .stButton>button:hover {
        background-position: right center;
        color: #fff;
        text-decoration: none;
        border: none;
    }

    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #1E3D59;
        border-radius: 12px;
        padding: 25px;
        background-color: #FAFAFA;
    }

    /* Container & Tab Styling */
    .st-emotion-cache-r421ms, .stTabs {
        border-radius: 12px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        padding: 15px;
    }
    
    /* Status Messages */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """
    Main function to run the Streamlit application.
    Orchestrates the data pipeline: Clean -> Report -> Visualize.
    """
    st.title("🚀 AI-Powered Data Science Pipeline")
    st.markdown("From Raw Data to Actionable Insights. Upload your data, provide instructions, and let a team of AI agents deliver a complete analysis.")

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.image("https://i.imgur.com/g0fGZ2Q.png", width=80)
        st.header("⚙️ Pipeline Configuration")
        
        uploaded_file = st.file_uploader(
            "1. Upload Your Data",
            type=["csv", "xlsx"],
            help="Upload a CSV or Excel file for analysis."
        )
        
        instructions = st.text_area(
            "2. Define Cleaning Instructions",
            height=150,
            placeholder="e.g., 'Handle missing sales values with the mean. Remove duplicates. Ensure OrderDate is a datetime.'"
        )
        
        start_button = st.button("✨ Run Full Pipeline")

    # --- Main Content Area for Data Preview and Reports ---
    col1, col2 = st.columns((2, 1))

    with col2:
        st.header("📊 Data Preview")
        preview_container = st.container(border=True, height=700)
        if uploaded_file is not None:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, nrows=100) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, nrows=100)
                preview_container.dataframe(df.head())
            except Exception as e:
                preview_container.error(f"Error reading file: {e}")
        else:
            preview_container.info("Upload a file to see a preview of your data here.")

    output_container = col1.container()
    
    if 'pipeline_started' not in st.session_state:
        st.session_state.pipeline_started = False

    if start_button or st.session_state.pipeline_started:
        if not uploaded_file or not instructions:
            st.warning("Please upload a file and provide instructions before starting.")
            return

        st.session_state.pipeline_started = True
        
        # Define temporary directories that will be cleaned up
        temp_dir = "temp_data"
        viz_dir = "visualizations"

        try:
            # --- Setup Temporary Environment ---
            # Clean up old directories first to ensure a fresh start
            if os.path.exists(viz_dir):
                shutil.rmtree(viz_dir)
            
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(viz_dir, exist_ok=True)
            
            uploaded_file.seek(0)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with output_container:
                status_area = st.empty()

                # --- STAGE 1: DATA CLEANING PIPELINE ---
                status_area.info("🚀 **Stage 1/3: Data Cleaning Agent is activating...**")
                
                cleaner_agent = DataAnalystAgent()
                graph = StateGraph(AgentStateModel)
                graph.add_node("supervisor", cleaner_agent.supervisor_node)
                graph.add_node("PreprocessingPlanner_node", cleaner_agent.PreprocessingPlanner_node)
                graph.add_node("Cleaner_node", cleaner_agent.Cleaner_node)
                graph.add_edge(START, "supervisor")
                cleaning_app = graph.compile()
                
                initial_state = AgentStateModel(
                    Instructions=instructions, Path=file_path, messages=[], Analysis=[], next="", current_reasoning=""
                )

                with st.spinner("Agent is analyzing and processing the data... This may take a moment."):
                    final_cleaning_state = cleaning_app.invoke(initial_state)

                if final_cleaning_state.get('next') != END:
                    status_area.error("❗️ **Data Cleaning Failed.** The agent could not process the file. Please check your instructions or data format.")
                    return
                
                st.balloons()
                status_area.success("✅ **Stage 1/3: Data Cleaning Complete!**")
                time.sleep(2)

                # --- STAGE 2: REPORTING PIPELINE ---
                status_area.info("🚀 **Stage 2/3: Reporting Agent is drafting the analysis...**")
                with st.spinner("Generating a comprehensive business report..."):
                    report_result = Report_agent(df_path=file_path)
                final_report = report_result.get("output", "Could not extract the final report.")
                status_area.success("✅ **Stage 2/3: Business Report Generated!**")
                time.sleep(2)

                # --- STAGE 3: VISUALIZATION PIPELINE ---
                status_area.info("🚀 **Stage 3/3: Visualization Agent is creating plots...**")
                with st.spinner("Creating insightful charts and graphs... This can be resource-intensive."):
                    viz_result = Visualizer_agent(df_path=file_path)
                viz_report_text = viz_result.get("output", "Could not extract visualization report.")
                status_area.success("✅ **Stage 3/3: Visualizations Created!**")
                time.sleep(2)
                
                status_area.success("🎉 **Pipeline Complete!** Your analysis is ready below.")

                # --- Display Final Outputs ---
                report_tab, viz_tab = st.tabs(["📈 Business Report", "📊 Visualizations"])

                with report_tab:
                    st.markdown(final_report)

                with viz_tab:
                    st.header("Generated Visualizations")
                    st.markdown("Each visualization is generated by the AI agent to highlight key trends, distributions, and comparisons in your data.")

                    # --- MODIFIED VISUALIZATION DISPLAY LOGIC ---
                    # Use a robust regex to find all file paths mentioned in the report
                    image_paths = re.findall(r"\(File:\s*(visualizations/[^)]+)\)", viz_report_text)

                    if not image_paths:
                        st.warning("The agent did not return any valid image paths. Displaying its raw output for debugging:")
                        st.text(viz_report_text)
                    else:
                        st.success(f"Successfully found {len(image_paths)} plots to display.")
                        
                        # Create columns for a grid layout, 2 plots per row
                        cols = st.columns(2)
                        col_idx = 0

                        for img_path in image_paths:
                            clean_img_path = img_path.strip().replace("\\", "/")
                            
                            with cols[col_idx % 2]:
                                if os.path.exists(clean_img_path):
                                    # Create a simple, clean title from the filename
                                    filename = os.path.basename(clean_img_path)
                                    title = filename.replace('_', ' ').replace('.png', '').title()
                                    
                                    with st.container(border=True):
                                        st.subheader(title)
                                        st.image(clean_img_path, use_column_width=True)
                                        st.caption(f"File: `{clean_img_path}`")
                                else:
                                    st.warning(f"Plot Not Found")
                                    st.error(f"The agent specified a plot at `{clean_img_path}`, but the file was not found on disk.")
                            
                            col_idx += 1


        except Exception as e:
            st.error("An unexpected pipeline error occurred. Please see the details below.")
            st.code(traceback.format_exc())
            
        finally:
            # --- AGGRESSIVE CLEANUP ---
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(viz_dir):
                shutil.rmtree(viz_dir)
            
            st.info("🧹 Temporary session files have been cleaned up.")
            st.session_state.pipeline_started = False

if __name__ == "__main__":
    main()