import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Langchain Classic imports for agent creation and execution
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor

# Google Gemini LLM integration
from langchain_google_genai import ChatGoogleGenerativeAI

# Langchain core prompt and output parser utilities
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from tools import search_tool

from typing import List, Optional

load_dotenv()  # Load environment variables from .env file


class ResearchResponse(BaseModel): #this is the structure of the output we want from the LLM
    topic: str
    summary: str
    sources: Optional[List[str]] = []
    tools_used: Optional[List[str]] = []

# set up an LLM using Gemini

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

print("Connecting to Gemini API (Starting Langchain agent)...")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

#here I am setting up the agent - pass llm langchain object
tools = [search_tool]
agent = create_tool_calling_agent(
    llm = llm,
    prompt=prompt,
    tools=tools
    )

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("What do you want to know about?: ")

raw_response = agent_executor.invoke({"input": query})
# print (raw_response)

# structured_response = parser.parse(raw_response["output"])
# print(structured_response)


if isinstance(raw_response["output"], list):
    output_text = raw_response["output"][0]["text"]
else:
    output_text = raw_response["output"].content if hasattr(raw_response["output"], "content") else str(raw_response["output"])

structured_response = parser.parse(output_text)
print(structured_response)