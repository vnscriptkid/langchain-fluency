import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.chains import LLMChain

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the language model
llm = ChatOpenAI(api_key=api_key)

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly and helpful AI assistant."),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

# Initialize the memory
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="messages"
)

# Create the LLMChain
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Chatbot: Goodbye! Have a great day!")
        break
    
    result = chain({"input": user_input})

    print(result["text"])
