import json
import asyncio
import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Make sure it's set in the .env file.")

llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

page = None

resume_text = """Name: Johnathan R. Smith
Phone: +91-XXXXXXXXXX | Email: johnsmith@email.com
 | LinkedIn: linkedin.com/in/johnsmith | Location: Bangalore, India

Executive Summary

Strategic and results-driven professional with 12+ years of experience in Technology Management, AI/ML Solutions, and Enterprise Software Development. Proven track record of leading cross-functional teams, delivering large-scale digital transformation projects, and driving business growth through innovative technology solutions. Adept at stakeholder management, process automation, and mentoring high-performance teams.

Core Competencies

AI/ML & Generative AI Solutions

Cloud Computing (AWS, Azure, GCP)

Enterprise Application Development

Project & Program Management (Agile/Scrum)

Stakeholder & Client Engagement

Strategic Roadmap Planning

Data Engineering & Analytics

Leadership & People Management

Professional Experience
Senior Engineering Manager – Infosys Ltd, Bangalore

Jan 2018 – Present

Spearheaded AI-driven digital transformation projects worth $10M+, improving client efficiency by 30%.

Directed a 40+ member engineering team across India, US, and Europe.

Designed and deployed a Generative AI-based HR Assistant handling 100k+ queries monthly with 95% accuracy.

Established cloud migration roadmap, moving legacy ERP systems to AWS with zero downtime.

Mentored mid-level managers and engineers, resulting in 20+ team members promoted internally.

Key Achievement:

Reduced project turnaround time by 25% by implementing Agile-Scaled frameworks across 5 business units.

Project Lead – Wipro Technologies, Hyderabad

Aug 2013 – Dec 2017

Led the development of enterprise AI chatbots and RPA solutions for banking & retail clients.

Implemented data preprocessing pipelines for large-scale analytics projects (~5TB datasets).

Coordinated with C-suite stakeholders to define KPIs, saving clients $2M annually.

Conducted regular training programs to upskill 100+ employees on AI/ML adoption.

Software Engineer – Tata Consultancy Services, Chennai

Jul 2010 – Jul 2013

Built scalable web applications serving 1M+ users across telecom and finance domains.

Improved system performance by 40% by optimizing backend algorithms.

Collaborated with product managers to translate business requirements into technical deliverables.

Education

MBA, Technology Management – IIM Bangalore (2017)

B.Tech, Computer Science – Anna University (2010)

Certifications

AWS Certified Solutions Architect – Professional

PMP® – Project Management Professional

DeepLearning.AI – Generative AI Specialization

Awards & Recognition

Infosys Excellence Award (2021): For leading enterprise-wide AI adoption.

Best Innovator (2016): Wipro Technologies for automation framework.

Publications & Speaking Engagements

Speaker at NASSCOM 2023 – “Agentic AI in Enterprise Solutions”

Published article in Analytics India Magazine – “RAG Systems for HR Automation”

Technical Skills

Languages: Python, Java, C++

Frameworks: LangChain, TensorFlow, PyTorch, FastAPI

Databases: PostgreSQL, MongoDB, Qdrant, Neo4j

Tools: Docker, Kubernetes, Git, Jenkins

References

Available on request
"""
resume_file_path = "john_doe_resume.txt"
with open(resume_file_path, "w") as f:
    f.write(resume_text)

# --- 3. TOOLS DEFINITION ---

@tool
async def scrape_website(url: str, headful: bool = False) -> dict:
    """
    Scrapes a website to extract job application form details.
    """
    print(f"Scraping URL: {url}...")
    # Helper functions...
    async def extract_label(page, el):
        try:
            if el_id := await el.get_attribute("id"):
                if label := await page.query_selector(f'label[for="{el_id}"]'):
                    if t := (await label.inner_text()).strip(): return t
            if aria := await el.get_attribute("aria-label"): return aria.strip()
            if pl := await el.get_attribute("placeholder"): return pl.strip()
            if prev := await page.evaluate("e => e.previousElementSibling?.innerText", el):
                if prev.strip(): return prev.strip()
        except: pass
        return None

    async def unique_xpath_for_element(page, handle):
        return await page.evaluate("""(e) => {
            function idx(n){let i=1,s=n.previousElementSibling;while(s){if(s.nodeName===n.nodeName)i++;s=s.previousElementSibling}return i}
            let seg='';while(e&&e.nodeType===1){let n=e.nodeName.toLowerCase(),i=idx(e);seg='/'+n+'['+i+']'+seg;e=e.parentElement}return seg;
        }""", handle)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headful)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            apply_button_selector = "text=/Apply for this job/i"
            print(f"Looking for the 'Apply' button with selector: '{apply_button_selector}'...")
            await page.wait_for_selector(apply_button_selector, state='visible', timeout=15000)
            await page.click(apply_button_selector)
            print("Successfully clicked the 'Apply' button.")
            form_ready_selector = "text=/resume/i"
            print(f"Waiting for form to be ready by looking for a reliable keyword: '{form_ready_selector}'...")
            await page.wait_for_selector(form_ready_selector, state='visible', timeout=10000)
            print("Application form is now visible and ready for scraping.")
        except Exception as e:
            await browser.close()
            return {"error": f"An unexpected error occurred during page interaction: {str(e)}"}
        
        forms_data = []
        form_container = page.locator(form_ready_selector).locator("xpath=ancestor::form").first
        if not await form_container.is_visible():
            form_container = page.locator(form_ready_selector).locator("xpath=ancestor::div[.//input or .//button]").first
        if await form_container.is_visible():
            controls_data = []
            elems = await form_container.locator("input, textarea, select, button, [role='button']").all()
            for el in elems:
                try:
                    if not (el_handle := await el.element_handle()): continue
                    tag = await el.evaluate("e => e.tagName.toLowerCase()")
                    xpath = await unique_xpath_for_element(page, el_handle)
                    controls_data.append({
                        "xpath": xpath, "tag": tag, "label_text": await extract_label(page, el),
                        "input_type": await el.get_attribute("type") or None,
                        "button_text": (await el.inner_text()).strip() if (tag == "button" or await el.get_attribute("role") == "button") else "",
                        "name_attr": await el.get_attribute("name"),
                        "required": await el.get_attribute("required") is not None,
                        "visible": await el.is_visible(),
                    })
                except Exception as e: print(f"Could not process an element: {e}")
            forms_data.append({"controls": controls_data})
        result = {"application_url": page.url, "forms": forms_data}
        await browser.close()
        print("Scraping finished successfully.")
        return result

@tool
async def fill_text_field(xpath: str, value: str) -> str:
    """Fills a text input field identified by its XPath with the provided value."""
    global page
    try:
        print(f"FILLING field at '{xpath}' with value '{value}'...")
        await page.locator(xpath).fill(value)
        return f"Successfully filled field at xpath {xpath}."
    except Exception as e: return f"Error filling field at xpath {xpath}: {e}"

@tool
async def upload_resume(xpath: str, file_path: str) -> str:
    """Uploads a file to a file input element identified by its XPath."""
    global page
    try:
        print(f"UPLOADING file '{file_path}' to input at '{xpath}'...")
        await page.locator(xpath).set_input_files(file_path)
        return f"Successfully set file input at {xpath} to '{file_path}'."
    except Exception as e: return f"Error uploading file at xpath {xpath}: {e}"

@tool
async def click_element(xpath: str) -> str:
    """Clicks an element on the page identified by its XPath (e.g., a submit button)."""
    global page
    try:
        print(f"CLICKING element at '{xpath}'...")
        await page.locator(xpath).click()
        return f"Successfully clicked element at xpath {xpath}."
    except Exception as e: return f"Error clicking element at xpath {xpath}: {e}"

# --- 4. AGENT DEFINITIONS ---

analyzer_system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Job Apply Agent. Your goal is to analyze a webpage. Use the `scrape_website` tool to get the form data from the URL."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
analyzer_agent = create_tool_calling_agent(llm_model, [scrape_website], analyzer_system_prompt)
analyzer_executor = AgentExecutor(agent=analyzer_agent, tools=[scrape_website], verbose=True, return_intermediate_steps=True)

filler_system_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert job application assistant. Your goal is to accurately fill out and submit a job application form.
You will be given:
1. A "form_data" JSON object which is a map of the application page, including the XPath for every field.
2. A "resume_data" JSON object containing the applicant's personal information.
3. The full text of the applicant's resume for context.
4. A file path for the applicant's resume file.
Your instructions are:
1. **Prioritize Resume Upload:** The absolute first step is to find the `input` field with `type='file'` and use the `upload_resume` tool.
2. **Fill Known Fields:** Go through each control in the `form_data.controls` list. For each, find the corresponding information in the `resume_data`. Use the `fill_text_field` tool for all text inputs.
3. **Generate Answers for Unknown Questions:** If you encounter a `textarea` for a question that is NOT in `resume_data`, you MUST generate a concise, professional answer (2-3 sentences) based on the provided resume context. Then, use `fill_text_field` to input your generated answer. DO NOT skip these fields if they are required.
4. **Handle Optional Fields:** For optional, non-essential fields like demographic questions (age, gender, ethnicity), you should skip them. Do not call any tools for these.
5. **Final Submission:** After all required fields are filled, find the control for the 'Submit Application' button and use the `click_element` tool to submit the form.
Think step-by-step. Announce which field you are filling before calling the tool.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
filler_tools = [fill_text_field, upload_resume, click_element]
filler_agent = create_tool_calling_agent(llm_model, filler_tools, filler_system_prompt)
filler_executor = AgentExecutor(agent=filler_agent, tools=filler_tools, verbose=True)

# --- 5. MAIN ORCHESTRATION LOGIC ---

async def main():
    job_url = "https://jobs.ashbyhq.com/ashby/81eb43b9-e8f1-412c-8b9f-3c81b377248d"

    print("--- PARSING RESUME ---")
    parsing_prompt = f"""
    Extract the following information from the resume text into a valid JSON object.
    Do NOT include any extra text, comments, or markdown formatting like ```json.
    Your entire response must be only the JSON object itself.
    - fullName
    - email
    - phone
    - linkedinURL

    Resume:
    {resume_text}
    """
    response = await llm_model.ainvoke(parsing_prompt)
    try:
        json_start = response.content.find('{')
        json_end = response.content.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            clean_json_str = response.content[json_start:json_end]
            resume_data = json.loads(clean_json_str)
            print("Resume parsed successfully:", resume_data)
        else: raise json.JSONDecodeError("Could not find JSON object in LLM response.", response.content, 0)
    except json.JSONDecodeError as e:
        print(f"Error parsing resume JSON from LLM response: {e}")
        print("Raw LLM response was:\n", response.content)
        return

    print("\n--- PHASE 1: ANALYZING JOB PAGE ---")
    analyzer_input = {"input": f"Scrape the website at the following URL: {job_url}"}
    analysis_result = await analyzer_executor.ainvoke(analyzer_input)
    
    # --- DEFINITIVELY CORRECTED FIX ---
    if 'intermediate_steps' in analysis_result and analysis_result['intermediate_steps']:
        # Get the last (action, observation) tuple from the list
        last_step_tuple = analysis_result['intermediate_steps'][-1]
        # The tool's output dictionary is the SECOND element (index 1) of the tuple
        tool_output_dict = last_step_tuple
    else:
        print("Analysis failed. No tool output found in intermediate steps.")
        return

    # Now, all checks are performed on the correctly extracted dictionary.
    if "error" in tool_output_dict:
        print(f"Analysis tool returned an error: {tool_output_dict['error']}")
        return
    if not tool_output_dict.get("forms") or not tool_output_dict["forms"]:
        print("Analysis failed. Could not find 'forms' in the tool output. Exiting.")
        return
    
    application_url = tool_output_dict.get("application_url", job_url)
    form_data = tool_output_dict["forms"]

    print("\n--- PHASE 2: FILLING APPLICATION ---")
    global page
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        print(f"Navigating to application page: {application_url}")
        await page.goto(application_url, wait_until="networkidle")
        await page.wait_for_timeout(2000)

        filler_task_prompt = f"""
        Here is the form data map:
        {json.dumps(form_data, indent=2)}

        Here is the applicant's resume data:
        {json.dumps(resume_data, indent=2)}

        Here is the full resume text for context on essay questions:
        ---
        {resume_text}
        ---

        The resume file is located at the local path:
        '{os.path.abspath(resume_file_path)}'

        Please fill out and submit the application based on these details.
        """
        await filler_executor.ainvoke({"input": filler_task_prompt})

        print("\nApplication process finished. Browser will close in 30 seconds.")
        await asyncio.sleep(30)
        await browser.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExecution stopped by user.")
