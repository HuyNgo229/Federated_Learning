import torch
from typing import List, Dict, Tuple
from .clients_on_server import ClientOnServer
import copy


class Server:
    def __init__(self):
        self.clients: List[ClientOnServer] = []
        self.server_model: Dict[str, torch.Tensor] = None
        self.loaded: bool = False
        self.client_num: int = 0
        self.client_weights_ratio: List[float] = []
        self.origin_model: Dict[str, torch.Tensor] = None

    def load_server_model(self, model_path: str):
      if not self.loaded:
        state_dict = torch.load(model_path)
        self.server_model.load_state_dict(state_dict)
        self.origin_model = copy.deepcopy(self.server_model)
        self.loaded = True


    def add_client(self, model_path: str, eval_list_path: str):
        client = ClientOnServer(model_path, eval_list_path)
        self.clients.append(client)
        self.client_num = len(self.clients)

    def __getitem__(self, index: int) -> ClientOnServer:
        return self.clients[index]

    def compute_client_samples_ratio(self) -> List[float]:
        total_samples = sum(client.get_samples() for client in self.clients)
        self.client_weights_ratio = [client.get_samples() / total_samples for client in self.clients]
        return self.client_weights_ratio

    def aggregate(self, args, models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            if args.mode.lower() == 'fedbn':