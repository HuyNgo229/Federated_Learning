import torch
import pickle
from typing import List, Dict, Tuple

class ClientOnServer:
    def __init__(self, model_path: str, eval_list_path: str):
        self.model_path = model_path
        self.eval_list_path = eval_list_path
        self.model = None
        self.eval_list = None
        self.load_model()
        self.load_eval_list()

    def load_model(self):
        self.model = torch.load(self.model_path)

    def load_eval_list(self):
        with open(self.eval_list_path, 'rb') as f:
            self.eval_list = pickle.load(f)

    def get_eval_list(self) -> List[Dict[str, float]]:
        return self.eval_list

    def get_model(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()
    
    def get_samples(self) -> int:
        return self.eval_list[0]['total_samples']
    
    def get_FL_round(self) -> int:
        return self.eval_list[0]['fl_round']

