# from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain.tools import tools
# from datetime import datetime

# search = DuckDuckGoSearchRun()
# search_tool = tools(
#     name="DuckDuckGo Search",
#     func=search.run,
#     description="Search the web for information",
# )



from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information using DuckDuckGo."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

search_tool = search_web  # Export this for main.py

