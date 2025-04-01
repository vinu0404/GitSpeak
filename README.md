# GitSpeak
(Live_Link)[https://gitspeak.onrender.com]

**GitSpeak** is an interactive Streamlit-based chatbot that lets you explore and query codebases from GitHub repositories. Powered by AWS Bedrock and Langchain, it supports Python (`.py`) and Jupyter Notebook (`.ipynb`) files, allowing users to ask technical questions about the code. With multi-chat functionality, and chat deletion options, GitSpeak is designed for developers, researchers, and anyone curious about code.


## Features

- **Chat with Codebases**: Load a GitHub repo and ask questions about its code (e.g., "What does `create_app` do? How to Improve the code ,Give code for that").
- **Multi-Chat Support**: Start new chats for different repositories, with each chat named after the repo .
- **Delete Chats**: Remove individual chats (and their vector stores) with a confirmation prompt, while keeping repo clones intact.
- **Supports `.py` and `.ipynb`**: Extracts functions and classes from Python scripts and Jupyter Notebooks using AST parsing.
- **Persistent Repo Storage**: Clones repos into a `repos/` folder with unique names, avoiding overwrites.
- **AWS Bedrock Integration**: Uses Bedrock embeddings and LLM for accurate, context-aware responses.

## Prerequisites

- **Python**: 3.10
- **AWS Account**: For Bedrock access (configure credentials via  AWS CLI).
- **Git**: For cloning repositories.

## AWS Configure
```
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
export AWS_REGION_NAME='your-region'  # e.g., us-east-1
```

##  Future Improvements
-  For querying languages link C,C++,Java and Javascript.
