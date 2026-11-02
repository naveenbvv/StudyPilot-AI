
import json

from ollama_client import ask_ai
from tools.study_tools import (
    calculate,
    create_study_plan,
    generate_notes
)


TOOLS = {
    "calculator": calculate,
    "study_planner": create_study_plan,
    "note_generator": generate_notes
}


def run_agent(user_request):

    tool_prompt = f"""
You are StudyPilot, an AI study assistant.

Choose the best tool for the user's request.

Available tools:
1. calculator: for mathematical calculations.
2. study_planner: for creating study schedules.
3. note_generator: for creating study notes.

Return ONLY valid JSON in this format:

For calculator:
{{"tool": "calculator", "arguments": {{"expression": "25*5"}}}}

For study planner:
{{"tool": "study_planner", "arguments": {{"subject": "Python", "days": 7}}}}

For notes:
{{"tool": "note_generator", "arguments": {{"topic": "Machine Learning"}}}}

If no tool is needed:
{{"tool": "none", "arguments": {{}}}}

User request: {user_request}
"""

    decision = ask_ai(tool_prompt)

    try:
        # Extract JSON decision
        start = decision.find("{")
        end = decision.rfind("}") + 1

        if start == -1 or end == 0:
            return "I couldn't understand the tool decision."

        action = json.loads(decision[start:end])

        tool_name = action["tool"]
        arguments = action["arguments"]

        if tool_name == "none":
            return ask_ai(user_request)

        if tool_name not in TOOLS:
            return "Unknown tool requested."

        # Execute selected tool
        result = TOOLS[tool_name](**arguments)

        # Ask AI to explain the tool result
        final_prompt = f"""
        User request: {user_request}

        Tool used: {tool_name}

        Tool result: {result}

        Explain the result clearly to the student.
        """

        return ask_ai(final_prompt)

    except Exception as error:
        return f"Agent error: {error}"


if __name__ == "__main__":

    print("===== StudyPilot AI =====")

    while True:

        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            break

        response = run_agent(question)

        print("\nStudyPilot:", response)







