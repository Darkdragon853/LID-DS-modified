from functools import lru_cache
from decimal import Decimal
import math
from pprint import pprint
import sys
import torch

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch import optim
from torch.utils.data import Dataset
from dataloader.syscall import Syscall
from algorithms.building_block import BuildingBlock
import numpy
import random

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class MLPDataset(Dataset):
    """
    torch dataloader that presents syscall data as tensors to neural network
    """

    def __init__(self, data):
        """
        adds all datapoints as tensors to dataset
        """
        self.x_data = []
        self.y_data = []
        for datapoint in data:
            self.x_data.append(torch.from_numpy(np.asarray(datapoint[0], dtype=np.float32)).to(device))
            self.y_data.append(torch.from_numpy(np.asarray(datapoint[1], dtype=np.float32)).to(device))

    def __len__(self):
        """
        returns the length of the dataset
        """
        return len(self.x_data)

    def __getitem__(self, index):
        """
        returns one item for a given index
        """
        _x = self.x_data[index]
        _y = self.y_data[index]

        return _x, _y


class MLP(BuildingBlock):
    """
        MLP Bulding Block built on pytorch
        initializes, trains and uses FeedForward Class from below

        Args:
            input_vector: the building block that is used for training
            output_label: the building block that is used for labeling the input vector
                            needs to be a vector with only one dimension != 0
            hidden_size: the number of neurons of the hidden layers
            hidden_layers: the number of hidden layers
            batch_size: number of input datapoints that are showed to the neural network
                            before adjusting the weights
    """

    def __init__(self,
                 input_vector: BuildingBlock,
                 output_label: BuildingBlock,
                 hidden_size: int,
                 hidden_layers: int,
                 batch_size: int,
                 learning_rate: float = 0.003,
                 use_independent_validation: bool = False
                 ):
        super().__init__()

        self.input_vector = input_vector
        self.output_label = output_label
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._use_independent_validation = use_independent_validation

        self._dependency_list = [input_vector, output_label]

        # estimated in train_on method
        self._input_size = 0
        self._output_size = 0

        self._training_set = set()
        self._validation_set = set()
        self._model = None  # to be initialized in fit()

        # number of epochs after which training is stopped if no improvement in loss has occurred
        self._early_stop_epochs = 500

        self._result_dict = {}
        self._train_counter = 0
    def train_on(self, syscall: Syscall):
        """
            building the training data set with input vector and labels
            estimating the input size

            Args:
                syscall: the current system call object
        """

        input_vector = self.input_vector.get_result(syscall)
        output_label = self.output_label.get_result(syscall)

        if input_vector is not None and output_label is not None:
            if self._input_size == 0:
                self._input_size = len(input_vector)

            if self._output_size == 0:
                self._output_size = len(output_label)
            self._training_set.add((input_vector, output_label))
            self._train_counter += 1

    def val_on(self, syscall: Syscall):
        """
            building the validation dataset

            Args:
                syscall: the current system call object
        """
        if self._use_independent_validation:
            pass
        else: 
            input_vector = self.input_vector.get_result(syscall)
            output_label = self.output_label.get_result(syscall)

            if input_vector is not None and output_label is not None:
                self._validation_set.add((input_vector, output_label))

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def fit(self):
        """
            trains the neural network
            initializes the FeedForward Model
            trains the model in batches using pytorch logic

            calculates loss on validation data and stops when no optimization occurs
        """
        print(f"MLP.train_set: {len(self._training_set)}".rjust(27))
        print(f"Using {self.learning_rate} as learning rate")
        if self._model is None:
            self._model = Feedforward(
                input_size=self._input_size,
                hidden_size=self.hidden_size,
                output_size=self._output_size,
                hidden_layers=self.hidden_layers
            ).model.to(device)
        self._model.train()

        criterion = nn.MSELoss()  # using mean squared error for loss calculation
        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)  # using Adam optimizer


        # Entscheidet ob die Sets so wie sie sind genommen werden oder nur das Trainingsset f??r den Fit benutzt wird. (Wird unterteilt in Training + Validation)
        if self._use_independent_validation:
            # building the datasets
            train_set_length = len(self._training_set)
            interrupt_counter = round(0.8 * train_set_length) # Aufteilung 80 % Training, 20% Verify
        
            # Sets for train-phase
            final_train_set = set()
            final_val_set = set()
        
            counter = 0
            for tuple in self._training_set:
                if counter >= interrupt_counter: 
                    final_val_set.add(tuple)
                else: 
                    final_train_set.add(tuple)
                counter += 1
        
            pprint(f"Train_set_length was {train_set_length}, is now splitted in {len(final_train_set)} parts training and {len(final_val_set)} parts verify.")

            train_data_set = MLPDataset(final_train_set) 
            val_data_set = MLPDataset(final_val_set)
        
        else: 
            # building the datasets
            train_data_set = MLPDataset(self._training_set) 
            val_data_set = MLPDataset(self._validation_set)
        
        # loss preparation for early stop of training
        epochs_since_last_best = 0
        best_avg_loss = math.inf
        best_weights = {}

        # Eliminate randomness
        generator = torch.Generator()
        generator.manual_seed(0)

        # initializing the torch dataloaders for training and validation
        train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=self.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=generator)
        val_data_loader = torch.utils.data.DataLoader(val_data_set, batch_size=self.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=generator)

        max_epochs = 50000
        # iterate through max epochs
        bar = tqdm(range(0, max_epochs), 'training'.rjust(27), unit=" epochs")  # fancy print for training        
        for e in bar:
            # training
            for i, data in enumerate(train_data_loader):
                inputs, labels = data

                optimizer.zero_grad()  # prepare gradients

                outputs = self._model(inputs)  # compute output
                loss = criterion(outputs, labels)  # compute loss

                loss.backward()  # compute gradients
                optimizer.step()  # update weights

            # validation
            val_loss = 0.0
            count = 0
            # calculate validation loss
            for i, data in enumerate(val_data_loader):
                inputs, labels = data
                #pprint(f"Input is: {inputs}, label is: {labels}")
                #pprint(len(labels[0])) # Irgendwie ist labels immer 183 lang - ungeachtet der BatchSize. Allerdings stimmt die Gr????e des Labels dann mit der L??nge vom OHE ein.
                #pprint(sum(0 == i for i in labels[0]))
                outputs = self._model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                count += 1
            avg_val_loss = val_loss / count

            if avg_val_loss < best_avg_loss:
                best_avg_loss = avg_val_loss
                best_weights = self._model.state_dict()
                epochs_since_last_best = 1
            else:
                epochs_since_last_best += 1

            # determine if loss optimization occurred in last x epochs, if not stop training
            stop_early = False
            if epochs_since_last_best >= self._early_stop_epochs:
                stop_early = True

            # refreshs the fancy printing
            # bar.set_description(f"fit MLP {epochs_since_last_best}|{best_avg_loss:.5f}".rjust(27), refresh=True)

            if stop_early:
                break
        
        print(f"stop at {bar.n} epochs with average loss of {best_avg_loss:.5f}".rjust(27))        
        self._result_dict = {}
        self._model.load_state_dict(best_weights)
        self._model.eval()
        
        # for param in self._model.parameters():
            # param.requires_grad = False
            # pprint(f"Model_param: {param}")


    def _calculate(self, syscall: Syscall):
        """ Forwards the anomaly calculation to the LRU-Cached implementation

        Args:
            syscall (Syscall): Current Syscall

        Returns:
            Anomaly-Value for this syscall
        """
        input_vector = self.input_vector.get_result(syscall)
        label = self.output_label.get_result(syscall)
        return self._cached_results(input_vector, label)

    @lru_cache(maxsize=1000)
    def _cached_results(self, input_vector, output_label):
        """
            calculates the anomaly score for one syscall
            idea: output of the neural network is a softmax layer containing the
                    estimated probability p for every possible output
                    1-p for the actual next syscall is then used as anomaly score

            Args:
                syscall: the current System Call Object

            returns: anomaly score
        """
        if input_vector is None:
            return None
        
        try:
            label_index = output_label.index(1) # getting the index of the actual next datapoint
        except ValueError:
            sys.exit(f'Unexpected ValueError in Output-Label. Please use an OHE. The label: {output_label}.Exiting.')
        
        in_tensor = torch.tensor(input_vector, dtype=torch.float32, device=device)
        with torch.no_grad():
            mlp_out = self._model(in_tensor)
        result = 1 - mlp_out[label_index].item()
        return Decimal(f'{result}')



    def depends_on(self):
        self.list = self._dependency_list
        return self.list

    def get_net_weights(self):
        """
            returns information about weights and biases of the neural network
        """

        # iterating over layers, if layer has weights it will be added with its index to the results
        weight_dict = {}
        for i in range(len(self._model)):
            if hasattr(self._model[i], 'weight'):
                weight_dict[str(i)] = {
                    'type': type(self._model[i]).__name__,
                    'in_features': self._model[i].in_features,
                    'out_features': self._model[i].out_features,
                    'weights': self._model[i].weight,
                    'bias': self._model[i].bias
                }
        return weight_dict
    
    def load_model(self, path: str):
        pass
    
    def save_model(self):
        if self._model is None:
            pprint('Model isn\'t available and therefore can\'t be saved.' )
            return
        
        pass
        

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class Feedforward:
    """
        handles the torch neural net by using the Sequential Class for mlp initialization
        implements adjustable hidden layers

        Args:
            input_size: the size of the input vector
            hidden_size: the number of neurons of the hidden layers
            output_size: the size of the output vector
            hidden_layers: the number of hidden layers
    """

    def __init__(self, input_size, hidden_size, output_size, hidden_layers):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        layer_list = self._get_mlp_sequence(hidden_layers)

        self.model = nn.Sequential(*layer_list)  # giving sequential the layers as list

    def _get_mlp_sequence(self, hidden_layers):
        """
            initializes the mlp layers as list
            number of hidden layers is adjustable

            input and hidden layers are Linear
            activation function is ReLU
            output layer is Softmax

            Args:
                number of hidden layers

        """
        hidden_layer_list = []
        for i in range(hidden_layers):
            hidden_layer_list.append(nn.Linear(self.hidden_size, self.hidden_size))
            hidden_layer_list.append(nn.Dropout(p=0.5))
            hidden_layer_list.append(nn.ReLU())


        return [
                   nn.Linear(self.input_size, self.hidden_size),
                   nn.Dropout(p=0.5),
                   nn.ReLU()
               ] + hidden_layer_list + [
                   nn.Linear(self.hidden_size, self.output_size),
                   nn.Dropout(p=0.5),
                   nn.Softmax(dim=0)
               ]
