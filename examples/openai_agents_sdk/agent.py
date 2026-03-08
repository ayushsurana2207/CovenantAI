import json
from openai import AsyncOpenAI

client = AsyncOpenAI()

# We construct mock classes to act as if they are output by the OpenAI Agents SDK stream parser.
class ToolCallItem:
    def __init__(self, name, arguments, call_id):
        self.name = name
        self.arguments = arguments
        self.id = call_id

class ToolCallOutputItem:
    def __init__(self, call_id, output):
        self.call_id = call_id
        self.output = output

class MessageOutputItem:
    def __init__(self, content):
        self.content = content

class RunResult:
    def __init__(self, new_items):
        self.new_items = new_items

# Real tool definitions
def search_web(query: str) -> str:
    return f"Search results for: {query}"

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny and 72F."

def confirm(action: str) -> str:
    return f"Action '{action}' confirmed by user."

def send_email(to: str, subject: str, body: str) -> str:
    return f"Email sent to {to} with subject '{subject}'."

class OpenAIAgentDriver:
    """A wrapper implementing a basic agent loop mapped exactly for OpenAIAgentsAdapter execution schemas."""
    def __init__(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a specific city.",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "confirm",
                    "description": "Ask the user for confirmation before performing a sensitive action like sending an email.",
                    "parameters": {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send an email. ALWAYS call confirm first before using this tool.",
                    "parameters": {"type": "object", "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"}
                    }, "required": ["to", "subject", "body"]}
                }
            }
        ]
        
    async def run(self, user_input: str) -> RunResult:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. You have access to tools for searching the web, checking the weather, and sending emails. IMPORTANT: Before sending an email, you must ALWAYS use the confirm tool first and wait for the response before invoking send_email."},
            {"role": "user", "content": user_input}
        ]
        items = []
        
        while True:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
            )
            msg = response.choices[0].message
            messages.append(msg)
            
            if msg.content:
                items.append(MessageOutputItem(msg.content))
                
            if not msg.tool_calls:
                break
                
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                items.append(ToolCallItem(tc.function.name, args, tc.id))
                
                # Execute tool
                tool_output = ""
                if tc.function.name == "search_web":
                    tool_output = search_web(**args)
                elif tc.function.name == "get_weather":
                    tool_output = get_weather(**args)
                elif tc.function.name == "confirm":
                    tool_output = confirm(**args)
                elif tc.function.name == "send_email":
                    tool_output = send_email(**args)
                    
                items.append(ToolCallOutputItem(tc.id, tool_output))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_output})
                
        return RunResult(items)

def get_agent():
    return OpenAIAgentDriver()
