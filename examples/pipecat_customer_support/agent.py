import asyncio
import json
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    TranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame
)

class MockCustomerSupportLLM(FrameProcessor):
    """
    Mock LLM mimicking a Pipecat Voice Agent handling Support calls.
    In a real application, this would be `OpenAILLMService` or `AnthropicLLMService`
    integrated directly into the Pipecat pipeline with Tools/Functions attached.
    """
    async def process_frame(self, frame, direction):
        await self.push_frame(frame, direction)
        
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
            text = frame.text.lower()
            
            # Simple matching for our YAML test scenarios
            if "about my account" in text and "john@example.com" in text:
                await self._push_tool_call("lookup_account", {"email": "john@example.com"}, "Found")
                await self._push_text("I have found your account, John.")
            
            elif "angry" in text or "real person" in text:
                await self._push_tool_call("transfer_to_human", {}, "Success")
                await self._push_text("I apologize for the frustration. I am transferring you to a representative now.")
                
            elif "double charged" in text:
                await self._push_tool_call("confirm_action", {"action": "refund"}, "Pending")
                await self._push_text("I can refund that for you. Please confirm you want me to process the refund.")
                
            elif "forgot my password" in text:
                await self._push_text("I can help with that. What is your email address?")
                
            elif text == "john@example.com":
                # Simulated continuation from "forgot my password"
                await self._push_tool_call("send_reset_link", {"email": "john@example.com"}, "Sent")
                await self._push_text("I have sent a password reset link to your email.")
                
            elif "wait, stop" in text:
                await self._push_text("I've stopped. How can I help?")
                
            elif "wife's account" in text:
                await self._push_text("I'm sorry, I am unauthorized to modify an account without the primary account holder present.")
                
            elif "internet is down" in text:
                await self._push_tool_call("search_kb", {"query": "internet down"}, "Article 1")
                await self._push_text("Let's troubleshoot. Have you tried restarting your router?")
                
            elif "restarted the router" in text:
                await self._push_tool_call("run_line_diagnostics", {}, "Outage Detected")
                await self._push_text("My diagnostics show a neighborhood outage. I will create a support ticket.")
                
            elif "neighborhood outage" in text:
                await self._push_tool_call("create_support_ticket", {"issue": "outage"}, "Ticket 123")
                await self._push_text("Ticket created. We will notify you when it is resolved.")
                
            elif "how do i go online" in text:
                await self._push_text("You can use cellular data on your phone.")
                
            elif "running out of data" in text:
                await self._push_tool_call("check_data_usage", {}, "High")
                await self._push_tool_call("suggest_premium_plan", {}, "Suggested")
                await self._push_text("It looks like you use a lot of data. Would you be interested in our premium unlimited plan?")
                
            elif "meaning of life" in text:
                await self._push_text("I can only help with customer support queries regarding your account.")
                
            elif "beep" in text:
                await self._push_tool_call("terminate_call", {}, "Ended")
                await self.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
                
            elif "cancel my subscription" in text:
                await self._push_tool_call("check_account_status", {}, "Active")
                await self._push_text("We're sorry to see you go. I can offer you a 50% discount for the next 3 months to stay.")
                
            elif "just want to cancel" in text:
                await self._push_tool_call("confirm_action", {"action": "cancel_subscription"}, "Pending")
                await self._push_text("Are you absolutely sure you want to cancel?")
                
            elif "yes, cancel it" in text:
                await self._push_tool_call("cancel_subscription", {}, "Canceled")
                await self._push_text("Your subscription has been canceled.")
                
            else:
                await self._push_text("I'm not sure how to help with that.")
                
    async def _push_tool_call(self, name, args, result):
        await self.push_frame(FunctionCallInProgressFrame(
            tool_call_id=f"call_{name}", function_name=name, arguments=json.dumps(args)
        ), FrameDirection.DOWNSTREAM)
        await self.push_frame(FunctionCallResultFrame(
            tool_call_id=f"call_{name}", function_name=name, arguments=json.dumps(args), result=result
        ), FrameDirection.DOWNSTREAM)
        
    async def _push_text(self, text):
        await self.push_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        await self.push_frame(LLMTextFrame(text), FrameDirection.DOWNSTREAM)
        await self.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

async def create_support_pipeline(transport):
    """
    CovenantAI executes this factory method directly via `covenant run`.
    It initializes the user's pipeline exactly as it would appear in production,
    except utilizing the injected `CovenantTestTransport` to observe traces.
    """
    return Pipeline([
        transport.input(),
        MockCustomerSupportLLM(),
        transport.output()
    ])
