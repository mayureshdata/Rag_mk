import os
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI 


# Load environment variables
load_dotenv()

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set your OpenAI API key in the .env file")

# Initialize the CSV Agent
def initialize_csv_agent(csv_file_path: str):
    """
    Initialize the CSV agent for querying the given CSV file.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        agent: A LangChain CSV agent object.
    """
    # Set up the LLM
    llm = ChatOpenAI(
        temperature=0,  # Low temperature for deterministic output
        # model="gpt-4",  # Use gpt-4 for better reasoning
        model="gpt-4o-mini",  # Use gpt-4 for better reasoning
        openai_api_key=api_key,
    )

    # Create the CSV Agent
    agent = create_csv_agent(llm=llm, path=csv_file_path, verbose=True,allow_dangerous_code=True)
    return agent


def main():
    """
    Main function to demonstrate querying a CSV file with the agent.
    """
    # Path to your CSV file
    csv_file = "healthcare_dataset.csv"  # Replace with your CSV file path

    # Initialize the agent
    try:
        agent = initialize_csv_agent(csv_file)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found.")
        return

    # Query the agent
    while True:
        query = input("\nEnter your query (type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting. Goodbye!")
            break

        try:
            # Get the response
            response = agent.invoke(query)
            # response = agent.run(query)
            print(f"Response:\n{response}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()


'''
pip install
langchain_experimental 
tabulate
'''