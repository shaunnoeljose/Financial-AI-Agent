from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv

load_dotenv()

# web search agent
web_search_agent = Agent(
    name = "Web Search Agent",
    role = "Search teh web for the information",
    model = Groq(id = "llama3-70b-8192"),
    tools = [DuckDuckGoTools()],
    instructions = ["Always include sources"],
    show_tool_calls=True,
    markdown = True
    )

# Financial Agent
finance_agent = Agent(
    name = "Financial AI Agent",
    model = Groq(id = "llama3-70b-8192"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                         company_news = True)],
    show_tool_calls=True,
    instructions = ["Use tables to display the data"],
    markdown = True 
)

multi_ai_agent = Agent(
    team = [web_search_agent, finance_agent],
    model=Groq(id="llama3-70b-8192"),
    instructions= ["Always include sources", "Use tables to display the data"],
    show_tool_calls = True,
    markdown = True
)
query = "Summarize analyst recommendation and share the latest top 5 news for NVDA"

multi_ai_agent.print_response(query, stream = True)


