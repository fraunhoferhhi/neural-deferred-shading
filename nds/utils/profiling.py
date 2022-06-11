import json
import time
import torch

class Profiler:
    def __init__(self, name=None, parent=None, device=None):
        """
        """
        self.device = device
        self.name = name
        self.parent = parent
        self.start_time = 0
        self.end_time = 0
        self.total = 0
        self.measurements = {}

    def start(self):
        self.start = time.perf_counter()

    def stop(self):
        # FIXME: Handle CPU
        torch.cuda.synchronize(self.device)
        self.end = time.perf_counter()
        self.total = self.end - self.start

        if self.parent:
            # Add own time
            self.parent.add_time(self.name, self.total)

            # Add child times
            for k, v in self.measurements.items():
                self.parent.add_measurement(f"{self.name}.{k}", v) 

    def current(self):
        return time.perf_counter() - self.start

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def add_time(self, name, time):
        self.add_measurement(name, {
                'count': 1,
                'total': time,
            })

    def add_measurement(self, name, measurement):
        if not name in self.measurements:
            self.measurements[name] = {
                'count': 0,
                'total': 0,
                'mean': 0
            }

        self.measurements[name]['count'] += measurement['count']
        self.measurements[name]['total'] += measurement['total']
        self.measurements[name]['mean'] = self.measurements[name]['total'] / self.measurements[name]['count']

    def record(self, name):
        return Profiler(name, parent=self, device=self.device)

    def export(self, path):
        with open(path, 'w') as f:
            json.dump({'total': self.total, 'measurements': self.measurements}, f, indent=4)

class NoOpProfiler:
    def start(self):
        pass

    def stop(self):
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def record(self, name):
        return NoOpProfiler()

    def export(self, path):
        pass