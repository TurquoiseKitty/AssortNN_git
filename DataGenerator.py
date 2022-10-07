# useful functions for synthetic data generation

import os
import yaml

from gen_models import TrueModel, AssortmentGenerator

import numpy as np
import random

import gen_models


def GenDataset(config_path = "gen_config/EXP1_MNL_20.yaml"):

    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)


    seed = params['Seed']
    np.random.seed(seed)
    random.seed(seed)

    name = params['DataSet']['name']
    N_sample = params['DataSet']['N_sample']
    path = params["DataSet"]["path"]

    AssortParams = params['Assortment']
    TrueModelParams = params['TrueModel']


    if not os.path.exists(path):
        os.makedirs(path)

    dir = os.listdir(path)
    if len(dir) > 0:
        print("Non empty directory")
        print("remove files!")
        for f in dir:
            os.remove(os.path.join(path, f))


    
    with open(path+"/Dataset_description.txt", 'w') as f:
        f.write("dataset name : "+name)
        f.write("\n")
        f.write("sample amount : "+str(N_sample))
        f.write("\n")
        f.write("config file : "+config_path)


    AssortmentGen = AssortmentGenerator.GenAssortment[AssortParams['scheme']]

    TrueModel = gen_models.TrueModel[TrueModelParams['model']](**TrueModelParams)

    TrueModel.save_para(path)

    N_prod = TrueModelParams['N_prod']


    INPUT = np.zeros((N_sample,N_prod))
    SAMP_OUTPUT = np.zeros((N_sample,N_prod))
    PROB_OUTPUT = np.zeros((N_sample,N_prod))

    for i in range(N_sample):
        
        assort = AssortmentGen(**AssortParams)

        # this part can be changed to other generating function
        
        INPUT[i] = assort

        SAMP_OUTPUT[i] = TrueModel.gen_final_choice(assort)
        
        PROB_OUTPUT[i] = TrueModel.prob_for_assortment(assort)



    np.save(path+"/"+"ASSORT",INPUT)
    np.save(path+"/"+"SAMP",SAMP_OUTPUT)
    np.save(path+"/"+"PROB",PROB_OUTPUT)

    print("dataset "+name + " generated!")



