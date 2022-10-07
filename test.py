import yaml
from models import ModelCollection
from models import Silly
from dataset import GrepFeatureDataset, GrepDataset
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from train import accuracy, KL_loss


def Loss_Plot(log_path = 'logs/main_exp'):

    train_loss = np.load(log_path+"/log_train_loss.npy")
    valid_loss = np.load(log_path+"/log_valid_loss.npy")

    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def Accu_Plot(log_path = 'logs/main_exp'):

    train_accu = np.load(log_path+"/log_train_accu.npy")
    valid_accu = np.load(log_path+"/log_valid_accu.npy")

    plt.plot(train_accu)
    plt.plot(valid_accu)
    plt.ylabel('accu')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def MNL_specialCheck(
    utility_file_path = "EXP1_datasets/MNL/NProd_20/utils.npy", 
    model_path = "logs/EXP1_MNL_20_VanillaMNL_LARGE/EXP1_MNL_20_VanillaMNL_LARGE_last.pth",
    N_prod = 20,
    gpu = True):

    print("actual probabilities:")
    utils = np.load(utility_file_path)

    probs = np.exp(utils)
    print(probs / sum(probs))

    print("predicted probs:")
    model = torch.load(model_path)


    input = torch.Tensor([[1]*N_prod, [1]*N_prod ])
    if gpu:
        input = input.cuda()

    print(model(input)[0])



def demo_check(training_yaml_path, model_name, model_path=""):

    with open(training_yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    if model_path == "":
        model_path = "logs/"+model_name+"/"+model_name+"_last.pth"

    group = params["exp_params"]["group"]
    if group == "EXP1":
        training_dataset, validating_dataset, testing_dataset = GrepDataset(**params['data_params'])


    model = torch.load(model_path)

    model.eval()

    IN, SAMP, PROB = testing_dataset[:]

    print("input :")
    print(IN[0:5])

    print("actual probs:")
    print(PROB[0:5])

    print("predicted probs:")
    OUT = model(IN)
    print(OUT[0:5])


def KL_loss_check(training_yaml_path, model_name, model_path="", datapath_overwrite=""):

    with open(training_yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    if model_path == "":
        model_path = "logs/"+model_name+"/"+model_name+"_last.pth"

    group = params["exp_params"]["group"]
    if group == "EXP1":

        if len(datapath_overwrite) > 0:
            params['data_params']['data_path'] = datapath_overwrite

        training_dataset, validating_dataset, testing_dataset = GrepDataset(**params['data_params'])


    model = torch.load(model_path)

    model.eval()

    IN, SAMP, PROB = testing_dataset[:]

    model_OUT = model(IN)

    best_OUT = PROB

    silly_model = Silly.Silly(PROB.shape[1])

    silly_OUT = silly_model(IN)

    print("Test KL loss: ",KL_loss(model_OUT, SAMP))
    print("Best possible KL loss: ",KL_loss(best_OUT, SAMP))
    print("Silly prediction KL loss: ",KL_loss(silly_OUT, SAMP))

