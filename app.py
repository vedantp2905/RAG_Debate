import os
import asyncio
import re
import shutil
import pandas as pd
import requests
import streamlit as st
from io import BytesIO
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew
from crewai_tools import PDFSearchTool
import tempfile

def verify_gemini_api_key(api_key):
    API_VERSION = 'v1'
    api_url = f"https://generativelanguage.googleapis.com/{API_VERSION}/models?key={api_key}"
    
    try:
        response = requests.get(api_url, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        return True
    except requests.exceptions.HTTPError:
        return False
    except requests.exceptions.RequestException as e:
        raise ValueError(f"An error occurred: {str(e)}")

def verify_gpt_api_key(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get("https://api.openai.com/v1/models", headers=headers)
    
    if response.status_code == 200:
        return True
    elif response.status_code == 401:
        return False
    else:
        print(f"Unexpected status code: {response.status_code}")
        return False

def process_content():
    sections = re.split(r'\*\*(.*?):\*\*', st.session_state.generated_content)
    proponent_lines = []
    opponent_lines = []

    for i in range(1, len(sections), 2):
        speaker = sections[i].strip()
        content = sections[i+1].strip()
        
        if "Proponent" in speaker:
            proponent_lines.append(content)
            if len(opponent_lines) < len(proponent_lines):
                opponent_lines.append("")
        elif "Opponent" in speaker:
            opponent_lines.append(content)
            if len(proponent_lines) < len(opponent_lines):
                proponent_lines.append("")

    max_length = max(len(proponent_lines), len(opponent_lines))
    proponent_lines += [""] * (max_length - len(proponent_lines))
    opponent_lines += [""] * (max_length - len(opponent_lines))

    data = {'Proponent': proponent_lines, 'Opponent': opponent_lines}
    st.session_state.df = pd.DataFrame(data)

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        st.session_state.df.to_excel(writer, index=False, sheet_name='Debate')
        workbook = writer.book
        worksheet = writer.sheets['Debate']
        
        for idx, col in enumerate(st.session_state.df.columns):
            series = st.session_state.df[col].dropna()
            max_len = max((
                series.astype(str).map(len).max(),
                len(col)
            )) + 2
            worksheet.set_column(idx, idx, max_len)
        
        wrap_format = workbook.add_format({'text_wrap': True})
        worksheet.set_column('A:B', None, wrap_format)

    excel_buffer.seek(0)
    st.session_state.excel_buffer = excel_buffer.getvalue()
            
def configure_tool(file_path):
    
    rag_tool = PDFSearchTool(
        pdf=file_path,
        config=dict(
            llm=dict(
                provider="openai",
                config=dict(
                    model="gpt-4o",
                    temperature=0.6
                ),
            ),

        )
    )
    
    return rag_tool

def generate_text(llm, rag_tool, topic, depth):
    inputs = {'topic': topic}

    manager = Agent(
        role='Debate Manager',
        goal='Ensure adherence to debate guidelines and format',
        backstory="""Experienced Debate Manager adept at overseeing structured debates
        across various domains. Skilled in maintaining decorum, managing time efficiently,
        and resolving unforeseen issues.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter = 5
    )

    proponent = Agent(
        role='Proponent of Topic',
        goal="""Present the most convincing arguments in favor of the topic,
        using only the information provided by the RAG tool.""",
        backstory="""You are an exceptional debater. Your task is to construct
        compelling arguments that support the given topic, but you must only use
        information retrieved from the provided documents.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[rag_tool],
        max_iter = 5

    )

    opposition = Agent(
        role='Opponent of Topic',
        goal="""Present the most convincing arguments against the topic,
        using only the information provided by the RAG tool.""",
        backstory="""You are a distinguished debater. Your task is to present
        well-rounded and thoughtful counterarguments, but you must only use
        information retrieved from the provided documents.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[rag_tool],
        max_iter = 5

    )

    writer = Agent(
        role='Debate Summarizer',
        goal="""Provide both sides' arguments, using only
        the information presented in the debate.""",
        backstory="""You are a highly respected journalist known for your impartiality.
        Your task is to synthesize the debate arguments, but you must not introduce
        any external information not presented by the debaters.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter = 5

    )

    task_manager = Task(
        description=f"""Manage the debate flow according to the specified format:
                       1- Both debaters must present short concise opening statements starting with the proponent
                       2- The debaters must rebuttal based on the output of their opponent starting with the proponent 
                       3- The total rebuttal rounds should be equal to: {depth}
                       4- The first rebuttal round should be based on opening statements of the debaters
                       5- Each subsequent rebuttal round must build on the previous rebuttal round
                       6- Each debater must give a short and concise closing argument
                       7- Ensure all arguments are based solely on information from the provided documents""",
        agent=manager,
        expected_output="A structured debate format with clear instructions for each stage of the debate, including opening statements, rebuttals, and closing arguments."
    )

    task_proponent = Task(
        description=f'''Research and present arguments supporting the topic: {topic}.
        Use ONLY the information retrieved by the RAG tool. Do not use external knowledge.''',
        agent=proponent,
        context=[task_manager],
        expected_output="A set of well-researched and compelling arguments supporting the topic, backed by evidence from the provided documents."
    )

    task_opposition = Task(
        description=f'''Research and present arguments opposing the topic: {topic}.
        Use ONLY the information retrieved by the RAG tool. Do not use external knowledge.''',
        agent=opposition,
        context=[task_manager],
        expected_output="A set of well-researched and compelling arguments opposing the topic, backed by evidence from the provided documents."
    )

    task_writer = Task(
        description="""Provide both sides' arguments, synthesizing key points,
        evidence, and rhetorical strategies into a cohesive report. Use only the information
        presented in the debate.""",
        agent=writer,
        context=[task_manager, task_proponent, task_opposition],
        expected_output="A script of the debate, highlighting the key arguments, evidence, and rhetorical strategies used by both sides."
    )

    crew = Crew(
        agents=[manager, proponent, opposition, writer],
        tasks=[task_manager, task_proponent, task_opposition, task_writer],
        verbose=2
    )

    result = crew.kickoff(inputs=inputs)
    return result
            
def main():
    
    st.header('Debate Generator')
    validity_model = False

    if 'generated_content' not in st.session_state:
        st.session_state.generated_content = None
    if 'topic' not in st.session_state:
        st.session_state.topic = ""
    if 'depth' not in st.session_state:
        st.session_state.depth = ""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'excel_buffer' not in st.session_state:
        st.session_state.excel_buffer = None
        
    with st.sidebar:
        with st.form('OpenAI'):
            api_key = st.text_input(f'Enter your OpenAI API key', type="password")
            submitted = st.form_submit_button("Submit")

        if api_key:
            validity_model = verify_gpt_api_key(api_key)
            
            if validity_model:
                st.write(f"Valid OpenAI API key")
            else:
                st.write(f"Invalid OpenAI API key")

    if validity_model:
        async def setup_OpenAI():
            loop = asyncio.get_event_loop()
            if loop is None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            os.environ["OPENAI_API_KEY"] = api_key
            llm = ChatOpenAI(model='gpt-4o', temperature=0.6, max_tokens=3000, api_key=api_key)
            print("OpenAI Configured")
            return llm

        llm = asyncio.run(setup_OpenAI())
        
        uploaded_file = st.file_uploader("Upload Kownledge Base PDF file", type="pdf")
        
        if uploaded_file:
        # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                rag_tool = configure_tool(tmp_file_path)
            finally:
            # Clean up the temporary file
                os.unlink(tmp_file_path)
        else:
            rag_tool = None       
        
        if uploaded_file:    
            topic = st.text_input("Enter the debate topic:", value=st.session_state.topic)
            depth = st.text_input("Enter the depth required:", value=st.session_state.depth)
            st.session_state.topic = topic
            st.session_state.depth = depth
        

        if st.button("Generate Content"):
            with st.spinner("Generating content..."):
                st.session_state.generated_content = generate_text(llm,rag_tool, topic, depth)
                process_content()

        if st.session_state.generated_content:
            st.markdown(st.session_state.generated_content)

            if st.session_state.excel_buffer is not None:
                st.download_button(
                    label="Download as Excel",
                    data=st.session_state.excel_buffer,
                    file_name=f"{st.session_state.topic}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
