
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"


def ask_ai(prompt):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        return result["message"]["content"]

    except requests.exceptions.RequestException as error:
        return f"Connection error: {error}"


if __name__ == "__main__":
    question = input("Ask StudyPilot AI: ")

    answer = ask_ai(question)

    print("\nStudyPilot AI:")
    print(answer)
