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
    return response.json()["items"], response.json()["has_more"]


def filter_questions_by_reason(questions, reasons, questions_per_reason):
    """Filters questions by the specified close reasons."""
    for question in questions:
        close_reason = question.get('closed_reason', '').lower()
        for reason in reasons:
            if close_reason == reason.lower():
                questions_per_reason["all-data"].append(question)


def save_questions_to_file(reason, questions):
    filename = f"data/raw/{reason.replace(' ', '_')}-raw-data.json"
    """Saves questions into separate files based on their close reason."""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(questions, file, ensure_ascii=False, indent=4)


def fetch_questions_for_reasons(reasons, limit_per_reason=1000):
    """Main function to fetch, filter, and save questions for each close reason."""
    questions_per_reason = {reason: [] for reason in reasons}
    page = 1

    while True:
        questions, has_more = fetch_questions(page, True)
        filter_questions_by_reason(questions, reasons, questions_per_reason)
        for reason, questions in questions_per_reason.items():

        if not has_more:
            break
        page += 1

        min_len = min([len(questions) for questions in questions_per_reason.values()])
        if min_len >= limit_per_reason:
            break
    for reason, questions in questions_per_reason.items():
        print(reason, len(questions))
    for reason, questions in questions_per_reason.items():
        save_questions_to_file(reason, questions)


def fetch_valid_questions(limit=3200):
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
    reasons = [
        "needs details or clarity",
        "needs more focus",
        "not suitable for this site",
        "opinion-based",
    ]
    fetch_questions_for_reasons(reasons)
    # fetch_valid_questions()


if __name__ == "__main__":
    main()
