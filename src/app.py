import streamlit as st
from streamlit_chat import message
import git
import os
import ast
import nbformat
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
import boto3
from dotenv import load_dotenv
load_dotenv()

# Custom CSS with Fixes
st.markdown("""
    <style>
    /* Main container and background */
    .chat-container { 
        height: calc(100vh - 180px); 
        overflow-y: auto; 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        border-radius: 15px; 
        padding: 15px; 
        background: linear-gradient(135deg, #2D3748, #4A5568); 
        backdrop-filter: blur(5px); 
        margin-top: 5px; 
    }
    /* User message bubble */
    .user-message { 
        background: linear-gradient(90deg, #38B2AC, #2C7A7B); 
        color: #FFFFFF; 
        border-radius: 15px 15px 0 15px; 
        padding: 12px; 
        margin: 10px 0; 
        max-width: 70%; 
        float: right; 
        clear: both; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.2); 
    }
    /* Bot message bubble */
    .bot-message { 
        background-color: #E2E8F0; 
        color: #1A202C; 
        border-radius: 15px 15px 15px 0; 
        padding: 12px; 
        margin: 10px 0; 
        max-width: 70%; 
        float: left; 
        clear: both; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.2); 
    }
    /* Input container */
    .input-container { 
        position: fixed; 
        bottom: 10px; 
        width: calc(90% - 20px); 
        max-width: 800px; 
        display: flex; 
        align-items: center; 
        gap: 10px; 
        background-color: #EDF2F7; 
        padding: 10px; 
        border-radius: 25px; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.15); 
    }
    /* Text input field */
    .stTextInput > div > div > input { 
        border-radius: 20px; 
        padding: 10px; 
        border: none; 
        background-color: #F7FAFC; 
        color: #1A202C; 
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1); 
        height: 38px; 
        line-height: 18px; 
        width: 100%; 
    }
    /* Hide the default input label */
    .stTextInput > label { 
        display: none; 
    }
    /* General buttons (e.g., Send, New Chat, Load Repository) */
    .stButton > button { 
        background: linear-gradient(90deg, #805AD5, #6B46C1); 
        color: #FFFFFF; 
        border-radius: 20px; 
        padding: 0 20px; 
        border: none; 
        height: 38px; 
        font-weight: bold; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.2); 
        line-height: 38px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
    }
    /* Title styling */
    .title { 
        font-size: 1.8em; 
        color: #4FD1C5; 
        text-shadow: 0 2px 4px rgba(0,0,0,0.3); 
        margin-bottom: 5px; 
    }
    /* Delete button */
    .delete-button { 
        background: linear-gradient(90deg, #F56565, #C53030); 
        color: #FFFFFF; 
        border-radius: 15px; 
        padding: 5px 10px; 
        border: none; 
        height: 30px; 
        font-size: 0.9em; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.2); 
    }
    /* Edit button */
    .edit-button { 
        background: linear-gradient(90deg, #ED8936, #DD6B20); 
        color: #FFFFFF; 
        border-radius: 15px; 
        padding: 5px 10px; 
        border: none; 
        height: 30px; 
        font-size: 0.9em; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.2); 
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] { 
        background-color: #2D3748; 
        color: #E2E8F0; 
    }
    /* Sidebar header */
    [data-testid="stSidebar"] .stHeader { 
        color: #4FD1C5; 
    }
    /* Hide Streamlit footer */
    [data-testid="stFooter"] { 
        display: none !important; 
    }
    /* Ensure full height to push footer off-screen if needed */
    html, body { 
        height: 100%; 
        margin: 0; 
        padding: 0; 
    }
    .stApp { 
        height: 100vh; 
        overflow: hidden; 
    }
    </style>
""", unsafe_allow_html=True)

# Initialize AWS Bedrock client with environment variables for Render
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME = os.environ.get("AWS_REGION_NAME", "us-east-1")  # Default to us-east-1 if not set

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    st.error("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables in Render.")
    st.stop()

bedrock_client = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION_NAME
)

# Step 1: Clone GitHub Repo into a unique folder
def clone_repo(repo_url):
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    local_path = os.path.join("repos", repo_name)
    os.makedirs("repos", exist_ok=True)
    if not os.path.exists(local_path):
        git.Repo.clone_from(repo_url, local_path)
    return local_path

# Step 2: Parse Code Structure (.py, .ipynb, .c, .cpp, .js)
def extract_code_units(file_path):
    units = []
    
    # Python (.py)
    if file_path.endswith(".py"):
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno
                    chunk = "\n".join(code.splitlines()[start_line-1:end_line])
                    metadata = {"file": file_path, "name": node.name, "type": type(node).__name__}
                    units.append((chunk, metadata))
        except SyntaxError:
            metadata = {"file": file_path, "name": "unknown", "type": "file"}
            units.append((code, metadata))
    
    # Jupyter Notebook (.ipynb)
    elif file_path.endswith(".ipynb"):
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                code = cell.source
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            chunk = code
                            metadata = {"file": file_path, "name": node.name, "type": type(node).__name__, "cell_index": notebook.cells.index(cell)}
                            units.append((chunk, metadata))
                except SyntaxError:
                    metadata = {"file": file_path, "name": "unknown", "type": "cell", "cell_index": notebook.cells.index(cell)}
                    units.append((code, metadata))
    
    # C, C++, JavaScript (.c, .cpp, .js)
    elif file_path.endswith((".c", ".cpp", ".js")):
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        metadata = {"file": file_path, "name": os.path.basename(file_path), "type": "file"}
        units.append((code, metadata))
    
    return units

# Step 3: Process Repo and Build Vector Store
@st.cache_resource
def build_vector_store(repo_url):
    repo_path = clone_repo(repo_url)
    code_units = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".py", ".ipynb", ".c", ".cpp", ".js")):
                file_path = os.path.join(root, file)
                units = extract_code_units(file_path)
                code_units.extend(units)
    if not code_units:
        return None
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        client=bedrock_client,
        model_kwargs={"dimensions": 512, "normalize": True}
    )
    chunks = [chunk for chunk, _ in code_units]
    metadata = [meta for _, meta in code_units]
    vector_store = LangchainFAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadata
    )
    return vector_store

# Step 4: Set Up LLM and Prompt
llm = ChatBedrock(
    model_id="amazon.nova-pro-v1:0",
    client=bedrock_client,
    model_kwargs={"temperature": 0.8, "max_tokens": 1000}
)

prompt_template = """You are an expert technical assistant designed to help users understand codebases and related technical concepts. Below is a user query followed by relevant code snippets retrieved from a repository. Your job is to provide a concise, accurate answer based on the snippets when possible, supplemented by your general technical knowledge if the snippets alone are insufficient. Do not invent details or assume functionality not present in the snippets unless it’s a reasonable inference grounded in standard programming practices. If the query has multiple parts (e.g., separated by a full stop or comma), address each part fully.

### Query:
{question}

### Retrieved Code Snippets:
{context}

### Instructions:
1. Analyze the code snippets to answer the query if they contain relevant information.
2. Provide a clear, technical explanation, referencing specific parts of the snippets (e.g., function names, key logic) when applicable.
3. If the query involves a function or class, describe its purpose, inputs, outputs, and key logic based on the snippets.
4. If the snippets lack sufficient information, use your general technical expertise to answer, explaining how the query relates to typical programming practices or the likely context of the codebase.
5. Keep the response focused, technical, and complete, addressing all parts of the query.
6. For questions unrelated to the codebase or technical topics (e.g., inappropriate or off-topic), respond with: 'Please ask a question related to the codebase or technical concerns.'
"""
prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)

def initialize_session(repo_name):
    return {
        "repo_url": "",
        "messages": [],
        "memory": ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True, output_key="answer"),
        "vector_store": None,
        "input_key": 0,
        "edit_index": None  # Tracks which message is being edited
    }

def get_unique_repo_name(repo_url, existing_names):
    base_name = repo_url.split('/')[-1].replace('.git', '')
    if base_name not in existing_names:
        return base_name
    counter = 1
    while f"{base_name}_{counter}" in existing_names:
        counter += 1
    return f"{base_name}_{counter}"

def main():
    # Initialize chats dictionary
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "delete_confirm" not in st.session_state:
        st.session_state.delete_confirm = None  # Tracks which chat to confirm deletion for

    # Sidebar for chat management
    with st.sidebar:
        st.header("Chat Sessions")
        if st.button("New Chat"):
            st.session_state.new_chat = True
            st.session_state.current_session_id = None

        # Display existing chats with delete buttons
        for session_id in list(st.session_state.chats.keys()):
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(session_id, key=f"select_{session_id}"):
                    st.session_state.current_session_id = session_id
                    st.session_state.new_chat = False
                    st.session_state.delete_confirm = None
            with col2:
                if st.button("🗑️", key=f"delete_{session_id}", help="Delete this chat", on_click=lambda sid=session_id: st.session_state.update({"delete_confirm": sid})):
                    pass

            # Confirmation dialog for deletion
            if st.session_state.delete_confirm == session_id:
                st.write(f"Are you sure you want to delete '{session_id}'?")
                col_confirm1, col_confirm2 = st.columns(2)
                with col_confirm1:
                    if st.button("OK", key=f"confirm_{session_id}"):
                        del st.session_state.chats[session_id]
                        if st.session_state.current_session_id == session_id:
                            st.session_state.current_session_id = None
                        st.session_state.delete_confirm = None
                        st.rerun()
                with col_confirm2:
                    if st.button("Cancel", key=f"cancel_{session_id}"):
                        st.session_state.delete_confirm = None
                        st.rerun()

    # Main UI
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("logo.png", width=50)
    with col2:
        st.markdown('<div class="title">GitSpeak</div>', unsafe_allow_html=True)

    # Handle new chat
    if "new_chat" in st.session_state and st.session_state.new_chat and st.session_state.current_session_id is None:
        st.write("Enter a GitHub repo URL for this new chat:")
        repo_url = st.text_input("GitHub Repo URL", key="new_repo_input", placeholder="e.g., https://github.com/pallets/flask.git")
        if st.button("Load Repository", key="load_new"):
            with st.spinner("Processing repository..."):
                repo_name = get_unique_repo_name(repo_url, st.session_state.chats.keys())
                st.session_state.chats[repo_name] = initialize_session(repo_name)
                vector_store = build_vector_store(repo_url)
                if vector_store:
                    st.session_state.chats[repo_name]["repo_url"] = repo_url
                    st.session_state.chats[repo_name]["vector_store"] = vector_store
                    st.session_state.chats[repo_name]["messages"] = [{"role": "assistant", "content": f"Loaded repo: {repo_name}. Ask me anything!"}]
                    st.session_state.current_session_id = repo_name
                    st.session_state.new_chat = False
                    st.success("Repository loaded successfully!")
                else:
                    st.warning("No code found. Chat disabled.")
                    del st.session_state.chats[repo_name]
            st.rerun()

    # Chat interface for current session
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.chats:
        current_session = st.session_state.chats[st.session_state.current_session_id]

        if current_session["vector_store"]:
            retriever = current_session["vector_store"].as_retriever(search_kwargs={"k": 6})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt},
                output_key="answer"
            )

            def run_qa(query):
                result = qa_chain({"query": query, "chat_history": current_session["memory"].load_memory_variables({})["chat_history"]})
                current_session["memory"].save_context({"query": query}, {"answer": result["answer"]})
                return result["answer"]

            # Display chat messages with edit functionality
            if current_session["messages"]:
                for i, msg in enumerate(current_session["messages"]):
                    if msg["role"] == "user":
                        # Display user message with Edit button
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            message(f"You: {msg['content']}", is_user=True, key=f"{st.session_state.current_session_id}_{i}_user", avatar_style=None)
                        with col2:
                            if st.button("Edit", key=f"edit_{st.session_state.current_session_id}_{i}", help="Edit this message"):
                                current_session["edit_index"] = i

                        # Edit mode for this message
                        if current_session["edit_index"] == i:
                            edited_text = st.text_input("Edit your message", value=msg["content"], key=f"edit_input_{i}")
                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                if st.button("Save", key=f"save_{i}"):
                                    old_content = current_session["messages"][i]["content"]
                                    current_session["messages"][i]["content"] = edited_text
                                    current_session["memory"].chat_memory.messages[i // 2 * 2].content = edited_text
                                    with st.spinner("Updating response..."):
                                        new_answer = run_qa(edited_text)
                                        current_session["messages"][i + 1]["content"] = new_answer
                                        current_session["memory"].chat_memory.messages[i // 2 * 2 + 1].content = new_answer
                                    current_session["edit_index"] = None
                                    st.rerun()
                            with col_cancel:
                                if st.button("Cancel", key=f"cancel_edit_{i}"):
                                    current_session["edit_index"] = None
                                    st.rerun()
                    else:
                        # Display assistant message without edit option
                        message(f"Bot: {msg['content']}", is_user=False, key=f"{st.session_state.current_session_id}_{i}_assistant", avatar_style=None)

            # Input for new messages
            with st.container():
                st.markdown('<div class="input-container">', unsafe_allow_html=True)
                col1, col2 = st.columns([4, 1])
                with col1:
                    user_input = st.text_input("", key=f"input_{st.session_state.current_session_id}_{current_session['input_key']}", placeholder="Ask a question...")
                with col2:
                    send_button = st.button("Send", key=f"send_{st.session_state.current_session_id}")
                st.markdown('</div>', unsafe_allow_html=True)

                if send_button and user_input:
                    with st.spinner("Generating answer..."):
                        answer = run_qa(user_input)
                        current_session["messages"].append({"role": "user", "content": user_input})
                        current_session["messages"].append({"role": "assistant", "content": answer})
                        current_session["input_key"] += 1
                    st.rerun()

    else:
        st.write("Start a new chat from the sidebar!")

if __name__ == "__main__":
    main()