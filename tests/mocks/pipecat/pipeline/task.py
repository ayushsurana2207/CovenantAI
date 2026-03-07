class PipelineParams:
    def __init__(self, **kwargs):
        pass

class PipelineTask:
    def __init__(self, pipeline, params=None):
        self.pipeline = pipeline
        self.params = params

    async def queue_frame(self, frame):
        if self.pipeline.processors:
            await self.pipeline.processors[0].process_frame(frame, direction=1)
