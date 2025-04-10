# GitSpeak
GitSpeak is  Streamlit-based chatbot that lets you explore and query codebases from GitHub repositories. Uses AWS Bedrock and Langchain, it supports Python (`.py`) and Jupyter Notebook (`.ipynb`) files, allowing users to ask technical questions about the code. Multi-chat functionality and it is designed for developers and researchers.


## Features

- **Chat with Codebases**: Load a GitHub repo and ask questions about its code (e.g., "What does `create_app` do? How to Improve the code ,Give code for that").
- **Multi-Chat Support**: Start new chats for different repositories, with each chat named after the repo .
- **Delete Chats**: Remove individual chats (and their vector stores) with a confirmation prompt, while keeping repo clones intact.
- **Supports `.py`,`.ipynb`,`.c`,`.c++` & `.js`**: Extracts functions and classes from  .c, c++ and .js scripts and for python and Jupyter Notebooks using AST parsing .
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