import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAI(api_key=api_key)

# Define the prompt template for generating code
code_prompt_template = PromptTemplate(
    input_variables=["language", "prompt"],
    template="Write {language} function that does the following: {prompt}"
)

# Define the prompt template for generating tests
test_prompt_template = PromptTemplate(
    input_variables=["language", "code"],
    template="Write runnable {language} test cases for the following code:\n\n{code}\n\nInclude import statements if necessary."
)

# Create the sequential chain using RunnableSequence
chain = (
    RunnablePassthrough.assign(
        code = code_prompt_template | llm | StrOutputParser()
    )
    | RunnablePassthrough.assign(
        tests = lambda x: test_prompt_template | llm | StrOutputParser()
    )
)

def generate_code_and_tests(language, prompt):
    result = chain.invoke({"language": language, "prompt": prompt})
    return result["code"], result["tests"]

# Example usage
language = "python"
prompt = "write python code that adds two numbers"
code, tests = generate_code_and_tests(language, prompt)

print("Generated Code:\n", code)
print("Generated Tests:\n", tests)
