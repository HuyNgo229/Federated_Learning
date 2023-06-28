import torch
import pickle
import argparse
from typing import List, Dict, Tuple
from server.clients_on_server import ClientOnServer
from server.server import Server




parser = argparse.ArgumentParser()
parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train and test')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--batch', type = int, default= 32, help ='batch size')
parser.add_argument('--round', type = int, default=100, help = 'Federated Learning for communication')
parser.add_argument('--epochs', type = int, default=1, help = 'epochs in local worker between communication')
parser.add_argument('--mode', type = str, default='fedbn', help='fedavg | fedbn')
args = parser.parse_args()

# Create instances of the classes and perform the operations
if __name__ == "__main__":
    server = Server()
    server.load_server_model('Server_model_path.pt')
    server.add_client('server\Model_from_Clients/model1_round1.pt', 'server\log/eval_list1.pickle')
    server.add_client('server\Model_from_Clients/model2_round1.pt', 'server\log/eval_list2.pickle')

    client_1: ClientOnServer = server[0]  # Access client 1 directly
    client_2: ClientOnServer = server[1]  # Access client 2 directly

    eval_list_1: List[Dict[str, float]] = client_1.get_eval_list()
    eval_list_2: List[Dict[str, float]] = client_2.get_eval_list()
    model_1: Dict[str, torch.Tensor] = client_1.get_model()
    model_2: Dict[str, torch.Tensor] = client_2.get_model()

    client_weights_ratio: List[float] = server.compute_client_samples_ratio()

    # Aggregate models
    aggregated_model: Dict[str, torch.Tensor] = server.aggregate(args, [model_1, model_2])

    # Compute average metrics
    train_loss, train_accuracy, test_loss, test_accuracy = server.metrics_avg()

    log_dict = {
        'fl_round': client_1.get_FL_round(),
        'training_loss_per_epoch': train_loss,
        'validation_loss_per_epoch': train_accuracy,
        'training_accuracy_per_epoch': test_loss,
        'validation_accuracy_per_epoch': test_accuracy,
    }

    fl_round = log_dict['fl_round']
    log_file_path = f"server/log_server/log_dict_round_{fl_round}.pickle"

    with open(log_file_path, 'wb') as f:
        pickle.dump(log_dict, f)

     # Save global model
    torch.save(aggregated_model, 'server/global_model.pt')
    print("Global model saved to global_model.pt")



    