import requests
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
access_token = os.getenv("STACK_ACCESS_TOKEN")
token = os.getenv("STACK_TOKEN")


def fetch_questions(page, closed, pagesize=100):
    """Fetches a page of closed questions from Stack Overflow."""
    url = "https://api.stackexchange.com/2.3/search/advanced"
    params = {
        "order": "desc",
        "sort": "creation",
        "site": "stackoverflow",
        "pagesize": pagesize,
        "page": page,
        "filter": "withbody",
        "closed": str(closed),
        # "access_token": access_token,  # Replace with your actual access token
        "key": token
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    print("PAGE FETCHED", page, response.json()["has_more"])
    return response.json()["items"], response.json()["has_more"]


def filter_questions_by_reason(questions, reasons, questions_per_reason, limit_per_reason):
    """Filters questions by the specified close reasons."""
    for question in questions:
        close_reason = question.get('closed_reason', '').lower()
        for reason in reasons:
            if close_reason == reason.lower() and len(questions_per_reason[reason]) < limit_per_reason:
                questions_per_reason[reason].append(question)


def save_questions_to_file(reason, questions):
    filename = f"data/raw/{reason.replace(' ', '_')}-raw-data.json"
    """Saves questions into separate files based on their close reason."""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(questions, file, ensure_ascii=False, indent=4)


def fetch_questions_for_reasons(reasons, limit_per_reason=300):
    """Main function to fetch, filter, and save questions for each close reason."""
    questions_per_reason = {reason: [] for reason in reasons}
    page = 1

    while any(len(questions) < limit_per_reason for questions in questions_per_reason.values()):
        questions, has_more = fetch_questions(page, True)
        filter_questions_by_reason(questions, reasons, questions_per_reason, limit_per_reason)
        for reason, questions in questions_per_reason.items():
            print("Reason: ", reason, "Question: ", len(questions))

        if not has_more:
            break
        page += 1

    for reason, questions in questions_per_reason.items():
        save_questions_to_file(reason, questions)


def fetch_valid_questions(limit=300):
    page = 100
    all_questions = []
    while len(all_questions) < limit:
        questions, has_more = fetch_questions(page, False)
        all_questions += questions
        print(len(all_questions))
        if not has_more:
            break
        page += 1

    save_questions_to_file("valid-questions", all_questions)
    return


def main():
    reasons = ["Needs details or clarity", "Needs more focus", "Opinion-based"]
    fetch_questions_for_reasons(reasons)
    # fetch_valid_questions()


if __name__ == "__main__":
    main()
