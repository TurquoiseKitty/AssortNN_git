import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd

def GrepDataset(
    data_seed,
    data_path,

    Assortments = "ASSORT.npy",
    Final_choices = "SAMP.npy",
    Real_prob = "PROB.npy",

    train_amount = 7000,
    valid_amount = 1000,
    test_amount = 1000,

    device = "gpu",

    **kwargs
):

    random.seed(data_seed)

    Assortments = np.load(data_path + "/" + Assortments)
    Final_choices = np.load(data_path + "/" + Final_choices)
    Real_prob = np.load(data_path + "/" + Real_prob)
   
    IN = torch.Tensor(Assortments)
    SAMP = torch.Tensor(Final_choices)
    PROB = torch.Tensor(Real_prob)


    if device == "gpu":
        IN = IN.to('cuda')
        SAMP = SAMP.to('cuda')
        PROB = PROB.to('cuda')


    total_data = len(IN)
    total_amount = train_amount + valid_amount + test_amount

    positions = random.sample(list(range(total_data)),k=total_amount)

    training_positions = positions[:train_amount]
    validating_positions = positions[train_amount:train_amount+valid_amount]
    testing_positions = positions[train_amount+valid_amount:total_amount]

    
    training_dataset = TensorDataset(
        IN[training_positions],
        SAMP[training_positions],
        PROB[training_positions]
    )
    
    validating_dataset = TensorDataset(
        IN[validating_positions],
        SAMP[validating_positions],
        PROB[validating_positions]
    )

    testing_dataset = TensorDataset(
        IN[testing_positions],
        SAMP[testing_positions],
        PROB[testing_positions]
    )
    
    

    return training_dataset, validating_dataset, testing_dataset

