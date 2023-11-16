import requests

def fetch_closed_questions(tag=None, limit=100):
    """
    Fetches closed questions from Stack Overflow using pagination.
    If a tag is provided, it fetches questions with that specific tag.
    'limit' determines the total number of questions to fetch.
    """
    url = "https://api.stackexchange.com/2.3/search/advanced"
    all_questions = []
    page = 1
    pagesize = 100  # Max allowed by API

    while len(all_questions) < limit:
        params = {
            "order": "desc",
            "sort": "creation",
            "site": "stackoverflow",
            "pagesize": pagesize,
            "page": page,
            "closed": "True"
        }
        if tag:
            params["tagged"] = tag

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        all_questions.extend(data["items"])

        if not data["has_more"]:
            break  # Stop if there are no more questions to fetch
        page += 1

    return all_questions[:limit]  # Return only the number of questions requested

def main():
    tag = input("Enter a tag to filter by (or leave blank for no filter): ").strip()
    limit = int(input("Enter the total number of questions to fetch: "))

    try:
        questions = fetch_closed_questions(tag, limit)
        for question in questions:
            print(f"Title: {question['title']}")
            print(f"Link: {question['link']}")
            print(f"Tags: {', '.join(question['tags'])}")
            print(f"Closed Reason: {question.get('closed_reason', 'Not available')}\n")
    except requests.RequestException as e:
        print(f"Error fetching questions: {e}")

if __name__ == "__main__":
    main()