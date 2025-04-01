import os
import ast
import git
import shutil
import time
import pandas as pd
import numpy as np
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import boto3
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download NLTK data (run once if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

# Initialize AWS Bedrock client
bedrock_client = boto3.client('bedrock-runtime')

# Clone GitHub Repo
def clone_repo(repo_url, local_path="repo", garbage_path="garbage"):
    if os.path.exists(local_path):
        try:
            if not os.path.exists(garbage_path):
                os.makedirs(garbage_path)
            new_garbage_path = os.path.join(garbage_path, f"repo_{int(time.time())}")
            shutil.move(local_path, new_garbage_path)
            print(f"Moved existing repo to {new_garbage_path}")
        except Exception as e:
            print(f"Failed to move repo to garbage: {e}")
            return None
    try:
        git.Repo.clone_from(repo_url, local_path)
        return local_path
    except git.GitCommandError as e:
        print(f"Git clone failed: {e}")
        return None

# Parse Code Structure with AST
def extract_code_units(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    try:
        tree = ast.parse(code)
        units = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                start_line = node.lineno
                end_line = node.end_lineno
                chunk = "\n".join(code.splitlines()[start_line-1:end_line])
                metadata = {"file": file_path, "name": node.name, "type": type(node).__name__}
                units.append((chunk, metadata))
        return units
    except SyntaxError:
        return []

# Build Vector Store
def build_vector_store(repo_url):
    repo_path = clone_repo(repo_url)
    if not repo_path:
        return None
    code_units = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                units = extract_code_units(file_path)
                code_units.extend(units)

    if not code_units:
        print("No Python code units found in the repository.")
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

# Set Up LLM and Prompt
llm = ChatBedrock(
    model_id="amazon.nova-pro-v1:0",
    client=bedrock_client,
    model_kwargs={"temperature": 0.8, "max_tokens": 1000}
)

prompt_template = """You are an expert code assistant tasked with answering questions about a codebase. Below is a user query followed by relevant code snippets retrieved from the repository. Your job is to analyze the snippets and provide a concise, accurate answer based solely on the provided code. Do not invent details or assume functionality not present in the snippets. If the snippets lack sufficient information, say so and suggest where to look. Answer each part of question if it has two parts before and after full stop or comma > answer both.

### Query:
{question}

### Retrieved Code Snippets:
{context}

### Instructions:
1. Analyze the code snippets to answer the query.
2. Explain the functionality in clear, technical language.
3. Reference specific parts of the snippets from the code repo (e.g., function names or key logic) to support your answer.
4. If the query involves a function or class, describe its purpose, inputs, outputs, and key logic.
5. Keep the response technical and focused on the query, but ensure it is complete and fully addresses the question.

### Answer:
"""
prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)

# Load Ground Truth Dataset
def load_ground_truth(file_path="ground_truth.txt"):
    questions = []
    ground_truths = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            if lines[i].startswith("Question:"):
                questions.append(lines[i].replace("Question:", "").strip())
                ground_truths.append(lines[i + 1].replace("Answer:", "").strip())
    return questions, ground_truths

# Custom Evaluation Metrics
def compute_bleu(reference, hypothesis):
    """Compute BLEU score between reference and hypothesis."""
    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    return sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25))

def compute_rouge(reference, hypothesis):
    """Compute ROUGE-L score between reference and hypothesis."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure

def compute_meteor(reference, hypothesis):
    """Compute METEOR score between pre-tokenized reference and hypothesis."""
    ref_tokens = word_tokenize(reference.lower()) 
    hyp_tokens = word_tokenize(hypothesis.lower()) 
    return meteor_score([ref_tokens], hyp_tokens)  

def compute_cosine_similarity(embeddings_model, text1, text2):
    """Compute cosine similarity using Bedrock embeddings."""
    emb1 = embeddings_model.embed_query(text1)
    emb2 = embeddings_model.embed_query(text2)
    return cosine_similarity([emb1], [emb2])[0][0]

# Main Evaluation Function
def evaluate_rag(repo_url="https://github.com/vinu0404/Movie-Review-Analyzer.git"):
    # Build vector store
    vector_store = build_vector_store(repo_url)
    if not vector_store:
        print("Failed to build vector store. Exiting.")
        return

    # Set up QA chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        output_key="answer"
    )

    # Load ground truth
    questions, ground_truths = load_ground_truth()
    generated_answers = []
    contexts = []

    # Generate answers
    for question in questions:
        result = qa_chain({"query": question})
        generated_answers.append(result["answer"])
        context = "\n".join([doc.page_content for doc in result["source_documents"]])
        contexts.append(context)

    # Prepare dataset
    data = {
        "question": questions,
        "answer": generated_answers,
        "context": contexts,
        "ground_truth": ground_truths
    }
    dataset = pd.DataFrame(data)
    print("Dataset prepared:")
    print(dataset.head())

    # Initialize Bedrock embeddings for cosine similarity
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        client=bedrock_client,
        model_kwargs={"dimensions": 512, "normalize": True}
    )

    # Compute metrics
    bleu_scores = []
    rouge_scores = []
    meteor_scores = []
    cosine_scores = []

    for gen_answer, gt_answer in zip(generated_answers, ground_truths):
        bleu = compute_bleu(gt_answer, gen_answer)
        rouge = compute_rouge(gt_answer, gen_answer)
        meteor = compute_meteor(gt_answer, gen_answer)
        cosine = compute_cosine_similarity(embeddings, gen_answer, gt_answer)

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        meteor_scores.append(meteor)
        cosine_scores.append(cosine)

    # Aggregate results
    avg_results = {
        "BLEU": np.mean(bleu_scores),
        "ROUGE-L": np.mean(rouge_scores),
        "METEOR": np.mean(meteor_scores),
        "Cosine Similarity": np.mean(cosine_scores)
    }

    # Output results
    print("Evaluation Results:")
    for metric, score in avg_results.items():
        print(f"{metric}: {score:.2f}")

    # Detailed results
    print("\nDetailed Scores:")
    for i, (q, gen, gt, bleu, rouge, meteor, cosine) in enumerate(zip(questions, generated_answers, ground_truths, bleu_scores, rouge_scores, meteor_scores, cosine_scores)):
        print(f"Q{i+1}: {q}")
        print(f"Generated: {gen}")
        print(f"Ground Truth: {gt}")
        print(f"BLEU: {bleu:.2f}, ROUGE-L: {rouge:.2f}, METEOR: {meteor:.2f}, Cosine Similarity: {cosine:.2f}\n")

    return avg_results

if __name__ == "__main__":
    evaluate_rag()