
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import re
from datetime import datetime, timedelta
import google.generativeai as genai


app = FastAPI()


# ---------- Pydantic Model ----------
class AnalyzeRequest(BaseModel):
    function_id: str
    repo_slug: str
    repo_token: str
    workspace: str
    description: str


# ---------- Helper: Get commits ----------
def get_commits(function_id, repo_slug, repo_token, workspace):
    API_URL = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}/commits"
    headers = {"Authorization": f"Bearer {repo_token}"}
    filtered_commits = []

    # Regular expression to match the valid format for function_id
    # Match lines like 'testcase: [TC1000, TC1001, TC1002]'
    function_id_pattern = re.compile(r"testcase\s*:\s*\[([^\]]+)\]", re.IGNORECASE)

    # Handle pagination
    next_url = API_URL
    first_connection = True
    while next_url:
        response = requests.get(next_url, headers=headers)
        if first_connection:
            if response.status_code == 200:
                print("Connection to Bitbucket made.")
            else:
                print(f"Failed to connect to Bitbucket. Status code: {response.status_code}")
                return response.status_code, []
            first_connection = False

        if response.status_code == 200:
            data = response.json()
            commits = data.get("values", [])
            for commit in commits:
                message = commit["message"].strip()
                match = function_id_pattern.search(message)
                if match:
                    # Extract all testcase IDs from the message
                    testcase_ids = [tc.strip() for tc in match.group(1).split(",")]
                    # Accept multiple function_ids as comma-separated string argument
                    filter_ids = [fid.strip() for fid in function_id.split(",")]
                    # If any of the filter_ids is present in the testcase_ids, include the commit
                    if any(fid in testcase_ids for fid in filter_ids):
                        print(f"Filtered commit message: {message}")
                        # Get code diff for this commit
                        diff_url = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}/diff/{commit['hash']}"
                        diff_response = requests.get(diff_url, headers=headers)
                        code_diff = diff_response.text if diff_response.status_code == 200 else None

                        commit_document = {
                            "hash": commit["hash"],
                            "message": message,
                            "author_username": commit["author"]["raw"].split("<")[0].strip(),
                            "author_email": commit["author"]["raw"].split("<")[1].strip().strip('>'),
                            "function_id": ', '.join([fid for fid in testcase_ids if fid in filter_ids]),
                            "code_diff": code_diff
                        }
                        filtered_commits.append(commit_document)
        next_url = data.get("next") if response.status_code == 200 else None

    if filtered_commits:
        return 200, filtered_commits
    else:
        return 404, []


# ---------- Helper: Call LLM ----------
def call_llm(filtered_commits, description):
    # Configure Gemini API key
    genai.configure(api_key="AIzaSyAeiO_ujdmWU0OGD6l2zxuSnzM1hT4SEl4")
    model = genai.GenerativeModel("gemini-2.5-flash")
    all_commit_info = []
    for commit in filtered_commits:
        code_diff = commit['code_diff']
        commit_info = f"Commit message: {commit['message']}\nAuthor: {commit['author_username']} <{commit['author_email']}>\nCode diff:\n{code_diff}"
        all_commit_info.append(commit_info)
    all_commit_info_str = "\n\n".join(all_commit_info)
    
    prompt = f"""
    Based on the above commit details, provide a single overall answer for the following:\nDoes the combined work in these commits align with the description above? Output only 'yes' or 'no'.\nHow much time would an industrial developer take for following code work? Output only a number.\nDescription: {description}\n\nCommit Details (all relevant commits):\n{all_commit_info_str}\n\n"""
    
    try:
        # Estimate token count before sending to LLM
        token_count = estimate_token_count(prompt)
        print(f"Estimated tokens sent to LLM: {token_count}")
        
        response = model.generate_content(prompt)
        ans = response.text.strip()
        print("\n--- LLM Output ---\n" + ans + "\n--- End LLM Output ---\n")
        # Parse the LLM output for progress, est_hours, and token count using regex for robustness
        import re
        progress = "None"
        est_hours = 0
        llm_token_count = None
        # Find all numbers in the output
        numbers = re.findall(r'\b(yes|no|\d+(?:\.\d+)?)\b', ans, re.IGNORECASE)
        # Try to extract progress, est_hours, and token count from the order of answers
        # 1. progress (yes/no), 2. est_hours (number), 3. token count (number)
        if numbers:
            # Find yes/no for progress
            for n in numbers:
                if n.lower() in ["yes", "no"]:
                    progress = n.lower()
                    break
            # Find numbers for est_hours and token count
            num_values = [float(n) for n in numbers if n.replace('.', '', 1).isdigit()]
            if len(num_values) >= 1:
                est_hours = num_values[0]
            if len(num_values) >= 2:
                llm_token_count = int(num_values[1])
        print(f"LLM-reported input token count: {llm_token_count if llm_token_count is not None else 'unknown'}")
        return 200, progress, est_hours
    except Exception as e:
        if 'context window' in str(e) or 'token' in str(e):
            print("Error: Input token limit exceeded. Please reduce the number or size of commits sent to the LLM.")
            return 413, "None", 0
        else:
            print(f"Error calling LLM: {e}")
            return 500, "None", 0

def estimate_token_count(text):
    # Rough estimate: 1 token ≈ 4 characters
    return len(text) // 4

# ---------- FastAPI Endpoint ----------
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    code_status, filtered_commits = get_commits(
        req.function_id, req.repo_slug, req.repo_token, req.workspace
    )

    if code_status == 404:
        raise HTTPException(status_code=404, detail="No commits found for the given function ID(s).")
    elif code_status != 200:
        raise HTTPException(status_code=code_status, detail="Failed to connect to Bitbucket.")

    llm_status, progress, est_hours = call_llm(filtered_commits, req.description)

    if llm_status == 413:
        raise HTTPException(status_code=413, detail="LLM input too large.")
    elif llm_status != 200:
        raise HTTPException(status_code=llm_status, detail="LLM processing failed.")

    return {
        "status": "success",
        "progress": progress,
        "estimated_hours": est_hours
    }
