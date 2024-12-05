import time

class Timer:
    def __init__(self):
        self.stack = []
        self.times = {}  # context_name -> cumulative time
        self.last_analysis_time = time.time()

    def time(self, name):
        return self.ContextManager(self, name)

    def analysis(self):
        total_time = time.time() - self.last_analysis_time
        return_dict = {}
        for name, t in self.times.items():
            return_dict[f"{name}:%"] = 100.0 * t / total_time if total_time > 0 else 0.0
            return_dict[f"{name}:t"] = t
        # Reset for next period
        self.times = {}
        self.last_analysis_time = time.time()
        return return_dict

    class ContextManager:
        def __init__(self, timer, name):
            self.timer = timer
            self.name = name

        def __enter__(self):
            self.start_time = time.time()
            self.child_time = 0.0
            self.timer.stack.append(self.name)
            self.context_name = "/".join(self.timer.stack)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            end_time = time.time()
            total_elapsed = end_time - self.start_time            
            self.timer.times[self.context_name] = self.timer.times.get(self.context_name, 0.0) + total_elapsed
            self.timer.stack.pop()
