class PipelineRunner:
    def __init__(self, **kwargs):
        pass

    async def run(self, task):
        if task.pipeline.processors:
            input_proc = task.pipeline.processors[0]
            for i in range(len(task.pipeline.processors) - 1):
                task.pipeline.processors[i].link(task.pipeline.processors[i+1])
            if hasattr(input_proc, "start"):
                await input_proc.start()
