# from dataclasses import dataclass
# from typing import Any
# from typing import List

# @dataclass
# class Metrics:
#     loss: List[Any]
#     rep_loss: List[Any]

class Metrics:
    def __init__(self, *args):
        # self.loss = []
        self.rep_loss = {
            m:[] for m in args
        }

def init(*args):
    global metrics
    metrics = Metrics(*args)

