import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import streamlit as st
import pandas as pd
import os
import time   
import shutil
import tempfile
import base64
import traceback
from langgraph.graph import START, StateGraph, END

# --- Import Agent Logic ---
# Assumes these are synchronous functions returning a dictionary with 'success' and structured data
from Cleaner_Agent import DataAnalystAgent, AgentStateModel
from Report_agent import Report_agent
from Visualizer_agent import Visualizer_agent

# --- Matplotlib Backend Fix ---
import matplotlib
matplotlib.use('Agg')

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for an Extremely Impressive and Cool UI ---
st.markdown("""
<style>
    /* Main App Background */
    body {
        color: #E0E0E0; /* Light grey text */
        background-color: #0F172A; /* Deep navy blue */
    }
    .main {
        background-color: #0F172A;
    }

    /* Page Title & Headers */
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        font-weight: bold;
        text-align: center;
    }
    h1 {
        color: #FFFFFF;
        text-shadow: 2px 2px 8px rgba(0, 255, 255, 0.5);
    }
    h3 {
        color: #A0AEC0; /* Lighter grey for subtitle */
    }

    /* Sidebar Styling */
    .st-sidebar {
        background-color: #1E293B; /* Slightly lighter navy */
        border-right: 2px solid #334155;
    }
    .st-sidebar h2 {
        color: #FFFFFF;
        text-align: left;
    }

    /* Start Button & Interactive Elements */
    .stButton>button {
        color: #FFFFFF;
        background-image: linear-gradient(45deg, #3B82F6 0%, #8B5CF6 100%);
        border: none;
        border-radius: 12px;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(59, 130, 246, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px 0 rgba(139, 92, 246, 0.5);
    }

    /* Card Layout for Content */
    .st-emotion-cache-r421ms { /* Streamlit's default container class */
        background-color: #1E293B;
        border: 2px solid transparent;
        border-image: linear-gradient(45deg, #3B82F6, #8B5CF6) 1;
        border-radius: 12px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.3);
        padding: 25px;
        transition: all 0.3s ease;
    }
    .st-emotion-cache-r421ms:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px 0 rgba(139, 92, 246, 0.4);
    }

    /* Custom Class for Empty State */
    .empty-state {
        text-align: center;
        padding: 40px;
        border: 2px dashed #334155;
        border-radius: 12px;
    }
    .empty-state h2 {
        color: #FFFFFF;
    }
    .empty-state p {
        color: #A0AEC0;
        font-size: 1.1rem;
    }

    /* Custom Class for Live Status Log */
    .status-log {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        font-family: 'Courier New', Courier, monospace;
        color: #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)


# --- SYNC HELPER FUNCTION ---
def run_report_and_viz_agents(df_path: str, output_dir: str):
    """
    Runs the Report and Visualizer agents sequentially.
    """
    report_result = Report_agent(df_path=df_path)
    viz_result = Visualizer_agent(df_path=df_path, output_dir=output_dir)
    return report_result, viz_result

# --- HELPER FUNCTIONS ---
def cleanup_session_files():
    """Deletes the temporary directory and clears associated session state keys."""
    if 'temp_dir_path' in st.session_state and st.session_state.temp_dir_path:
        temp_dir = st.session_state.temp_dir_path
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error removing temp directory {temp_dir}: {e}")
    
    # Extended list of keys to clear for a full reset
    keys_to_clear = [
        'temp_dir_path', 'pipeline_run_complete', 
        'final_report_structured', 'final_visuals_structured'
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

@st.cache_data
def get_image_as_base64(path):
    """Reads an image file and returns its Base64 encoded string."""
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def display_empty_state():
    """Shows a visually appealing message when no file is uploaded."""
    st.markdown(
        """
        <div class="empty-state">
            <h2>Welcome to the AI Data Analyst</h2>
            <p>Upload your data and provide instructions in the sidebar to begin.</p>
            <p>Let's turn your raw data into stunning insights! ✨</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- MAIN APP ---
def main():
    # --- HEADER ---
    st.title("🤖 AI Data Analyst")
    st.markdown("<h3>Derive actionable insights from raw data in minutes from a specialized team of AI agents</h3>", unsafe_allow_html=True)
    st.write("")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Pipeline Configuration")
        uploaded_file = st.file_uploader("1. Upload Your Data File", type=["csv", "xlsx"])
        instructions = st.text_area("2. Describe Your Analysis Goal", height=150, placeholder="e.g., 'Analyze monthly sales trends and identify top-performing products.'")
        
        col1, col2 = st.columns(2)
        start_button = col1.button("✨ Run Analysis", type="primary")
        if col2.button("🧹 New Analysis"):
            cleanup_session_files()
            st.success("Session cleared.")
            time.sleep(1)
            st.rerun()

    # --- MAIN CONTENT AREA ---
    # Display empty state if no file is uploaded.
    if not uploaded_file:
        display_empty_state()
        return

    # Show data preview if a file is uploaded.
    with st.expander("📊 **View Data Preview**", expanded=False):
        try:
            uploaded_file.seek(0)
            df_preview = pd.read_csv(uploaded_file, nrows=100) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, nrows=100)
            st.dataframe(df_preview, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read the file preview. Error: {e}")


    # --- PIPELINE EXECUTION ---
    if start_button:
        if not instructions:
            st.warning("Please describe your analysis goal before starting.")
            return

        # Clean up previous session and set up a new one
        cleanup_session_files()
        st.session_state.temp_dir_path = tempfile.mkdtemp().replace('\\', '/')
        temp_file_path = os.path.join(st.session_state.temp_dir_path, uploaded_file.name).replace('\\', '/')
        
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # UI container for live logs
            log_container = st.container()
            with log_container:
                st.subheader("🤖 Agent Status Log")
                status_log = st.empty()
                log_messages = ["[INITIALIZING] Pipeline started..."]
                status_log.markdown(f"<div class='status-log'>{'<br>'.join(log_messages)}</div>", unsafe_allow_html=True)
                
                # --- STAGE 1: DATA CLEANING ---
                log_messages.append("🚀 **Stage 1/3:** Data Cleaning Agent activated...")
                status_log.markdown(f"<div class='status-log'>{'<br>'.join(log_messages)}</div>", unsafe_allow_html=True)
                with st.spinner("Agent is analyzing and cleaning the data..."):
                    cleaner_agent = DataAnalystAgent()
                    graph = StateGraph(AgentStateModel)
                    graph.add_node("supervisor", cleaner_agent.supervisor_node)
                    graph.add_node("PreprocessingPlanner_node", cleaner_agent.PreprocessingPlanner_node)
                    graph.add_node("Cleaner_node", cleaner_agent.Cleaner_node)
                    graph.add_edge(START, "supervisor")
                    cleaning_app = graph.compile()
                    initial_state = AgentStateModel(Instructions=instructions, Path=temp_file_path, messages=[], Analysis=[])
                    final_cleaning_state = cleaning_app.invoke(initial_state)

                    if final_cleaning_state.get('next') != END:
                        st.error("❗️ **Data Cleaning Failed.** Please check instructions or data.")
                        cleanup_session_files()
                        return
                
                log_messages.append("✅ **Stage 1/3:** Data Cleaning Complete!")
                status_log.markdown(f"<div class='status-log'>{'<br>'.join(log_messages)}</div>", unsafe_allow_html=True)
                st.balloons()
                
                # --- STAGES 2 & 3: REPORTING & VISUALIZATION ---
                log_messages.append("🚀 **Stages 2 & 3:** Reporting and Visualization agents activated...")
                status_log.markdown(f"<div class='status-log'>{'<br>'.join(log_messages)}</div>", unsafe_allow_html=True)
                with st.spinner("AI agents are generating the report and plots..."):
                    report_result, viz_result = run_report_and_viz_agents(
                        df_path=temp_file_path,
                        output_dir=st.session_state.temp_dir_path
                    )
                
                # Process and store results in session state
                if report_result and report_result.get("success"):
                    st.session_state.final_report_structured = report_result.get("parsed_report")
                else:
                    st.error(f"Report generation failed: {report_result.get('error', 'Unknown error')}")

                if viz_result and viz_result.get("success"):
                    st.session_state.final_visuals_structured = viz_result.get("parsed_visuals")
                else:
                    st.error(f"Visualization generation failed: {viz_result.get('error', 'Unknown error')}")

                # Final log update
                if st.session_state.final_report_structured and st.session_state.final_visuals_structured:
                    log_messages.append("✅ **Stages 2 & 3:** Report and Visualizations Complete!")
                    log_messages.append("🎉 **Pipeline Complete!** Displaying results below.")
                    st.session_state.pipeline_run_complete = True
                else:
                    log_messages.append("❗️ **PIPELINE FAILED:** One or more agents failed. Check error messages above.")
                
                status_log.markdown(f"<div class='status-log'>{'<br>'.join(log_messages)}</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error("An unexpected pipeline error occurred.")
            st.code(traceback.format_exc())
            cleanup_session_files()
            return
        
        # Rerun to display results from session state
        st.rerun()

    # --- DISPLAY RESULTS (persisted in session state) ---
    if st.session_state.get("pipeline_run_complete"):
        st.write("---")
        st.header("✨ Analysis Results")

        # Display the structured report
        if st.session_state.get("final_report_structured"):
            report_data = st.session_state.final_report_structured
            with st.container(border=True):
                st.subheader(report_data.get("subject", "Business Report"))
                
                # Use columns for a better summary layout
                col1, col2 = st.columns(2)
                with col1:
                    st.info("Executive Summary")
                    st.markdown(report_data.get("executive_summary", "Not available."))
                with col2:
                    st.info("💡 Biggest Strategic Opportunity")
                    st.markdown(report_data.get("strategic_opportunity", "Not available."))

                st.info("🔑 Key Insights & Patterns")
                st.markdown(report_data.get("key_insights_and_patterns", "Not available."))

                with st.expander("View Full Detailed Report"):
                    st.markdown("---")
                    st.subheader("Data Overview and Quality Review")
                    st.markdown(report_data.get("data_overview_and_quality_review", "Not available."))
                    st.markdown("---")
                    st.subheader("Descriptive and Diagnostic Analysis")
                    st.markdown(report_data.get("descriptive_and_diagnostic_analysis", "Not available."))
                    st.markdown("---")
                    st.subheader("Recommendations and Forecast")
                    st.markdown(report_data.get("recommendations_and_forecast", "Not available."))

        # Display the visualizations
        if st.session_state.get("final_visuals_structured"):
            visuals_data = st.session_state.final_visuals_structured
            st.write("")
            with st.container(border=True):
                st.subheader(visuals_data.get("report_title", "Generated Visualizations"))
                visualizations = visuals_data.get("visualizations", [])
                
                if not visualizations:
                    st.warning("The visualization agent did not return any visuals.")
                else:
                    # Create a grid layout for visualizations
                    cols = st.columns(2)
                    col_idx = 0
                    for vis in visualizations:
                        with cols[col_idx % 2]:
                            try:
                                st.subheader(vis.get("title", "Untitled Chart"))
                                image_path = vis.get("file_path")
                                if image_path and os.path.exists(image_path):
                                    st.image(image_path, use_column_width=True)
                                    st.markdown(f"**Insight:** {vis.get('insight', 'No insight provided.')}")
                                    st.caption(f"File: {os.path.basename(image_path)}")
                                    st.write("---")
                                else:
                                    st.warning(f"Chart image not found at path: {image_path}")
                            except Exception as e:
                                st.error(f"Could not display visual '{vis.get('title')}': {e}")
                        col_idx += 1

if __name__ == "__main__":
    main()



