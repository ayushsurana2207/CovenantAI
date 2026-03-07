from enum import Enum

class FrameDirection(Enum):
    DOWNSTREAM = 1
    UPSTREAM = 2

class FrameProcessor:
    def __init__(self):
        self._prev = None
        self._next = None

    def link(self, next_processor):
        self._next = next_processor
        next_processor._prev = self

    async def process_frame(self, frame, direction):
        pass

    async def push_frame(self, frame, direction):
        if direction == FrameDirection.DOWNSTREAM and self._next:
            await self._next.process_frame(frame, direction)
        elif direction == FrameDirection.UPSTREAM and self._prev:
            await self._prev.process_frame(frame, direction)
