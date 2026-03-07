# Minimal LangChain Agent
from langchain_core.runnables import RunnableLambda

def dummy_agent(user_input: str) -> str:
    # TODO: Replace with your actual LangChain agent
    if "weather" in user_input.lower():
         return "I would use the search tool here."
    if "bomb" in user_input.lower():
         return "I cannot do that."
    return "I am a helpful agent."

agent = RunnableLambda(dummy_agent)
