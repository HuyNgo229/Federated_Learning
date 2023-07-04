import os
import torch
from client import Client, eval_list
import pickle
from utils.model import create_cnn_model
import copy
""" 
Input of clients inclule:
1- file.pt from client\Model_from_Server
2- Custom Dict from argprase to config (batch_size,epochs,percentage_of_dataset, mode,dataset_name....)

Output of clients inclule:
1- file.pt from client\Model_Client_update
2- eval_list: Dict from  client\log
 """


""" Start Client """

data_dir = "dataset\histopathological_breast_cancer_dataset" ### path of dataset
batch_size = 32
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
percentage_of_dataset = 0.1
dataset_name = "histopathological_breast_cancer_dataset"
mode = "fedbn"
server_model_path = "client\Model_from_Server/model.pt" ## path of model from server
client_model_path = "client\Model_Client_update/model_client.pt" ## path of model client will save and send to server

if __name__ == "__main__":
    # Initalize client instance
    client = Client(
        data_dir=data_dir,
        batch_size=batch_size,
        epoch=epochs,
        device=device,
        model=None,  # Khởi tạo với None, sẽ được gán sau
        percentage_of_dataset=percentage_of_dataset,
        dataset_name=dataset_name,
        mode=mode,
        server_model_path=server_model_path,
        client_model_path=client_model_path
    )

    # Load data
    trainloader, validloader, num_examples = client.load_datasets()

    # # Model architecture from utils\model.py
    CNNModel = create_cnn_model()
    client.model = copy.deepcopy(CNNModel)

    # Fit model
    client.fit()

    # Evaluate model
    client.evaluate()

    # Save eval_list to a file
    with open('eval_list.pkl', 'wb') as f:
        pickle.dump(eval_list, f)
