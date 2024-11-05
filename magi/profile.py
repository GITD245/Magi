import torch

class magi_profile():
    def __init__(self,layers,local_expert_count,global_expert_count,window_size=10):
        self.layers = layers
        self.window_size = window_size
        pass
    def recode_local_expert_count(self,local_expert_count):
        pass
    def recode_global_expert_count(self,global_expert_count):
        pass
    def recode_time(self):
        pass
    def all_gather(self):
        pass