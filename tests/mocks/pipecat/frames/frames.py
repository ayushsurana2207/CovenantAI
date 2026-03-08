class Frame:
    pass

class TranscriptionFrame(Frame):
    def __init__(self, text="", user_id="", timestamp=""):
        self.text = text
        self.user_id = user_id
        self.timestamp = timestamp

class LLMFullResponseStartFrame(Frame):
    pass


class LLMFullResponseEndFrame(Frame):
    pass

class LLMTextFrame(Frame):
    def __init__(self, text):
        self.text = text

class FunctionCallInProgressFrame(Frame):
    def __init__(self, tool_call_id, function_name, arguments):
        self.tool_call_id = tool_call_id
        self.function_name = function_name
        self.arguments = arguments

class FunctionCallResultFrame(Frame):
    def __init__(self, tool_call_id, function_name, arguments, result):
        self.tool_call_id = tool_call_id
        self.function_name = function_name
        self.arguments = arguments
        self.result = result

class ErrorFrame(Frame):
    def __init__(self, error):
        self.error = error

class StartFrame(Frame):
    pass


class EndFrame(Frame):
    pass


class InterruptionFrame(Frame):
    pass

class TTSAudioRawFrame(Frame):
    def __init__(self, audio, sample_rate, num_channels):
        self.audio = audio
        self.sample_rate = sample_rate
        self.num_channels = num_channels
