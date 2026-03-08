from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def get_weather(city: str) -> str:
    """Get the weather for a specific city."""
    return f"The weather in {city} is sunny and 72F."

@tool
def confirm(action: str) -> str:
    """Ask the user for confirmation before performing a sensitive action like sending an email."""
    # In a real agent, this would pause and ask the human. 
    # For testing, we just return that it was confirmed.
    return f"Action '{action}' confirmed by user."

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. ALWAYS call confirm first before using this tool."""
    return f"Email sent to {to} with subject '{subject}'."

def get_agent() -> AgentExecutor:
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Define the tools
    tools = [search_web, get_weather, confirm, send_email]
    
    # Define the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You have access to tools for searching the web, checking the weather, and sending emails. IMPORTANT: Before sending an email, you must ALWAYS use the confirm tool first and wait for the response before invoking send_email."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor
