from train import trainer
import yaml
from models import ModelCollection
from dataset import GrepDataset
from torch.utils.data import DataLoader
import numpy as np


# run in jupyter notebook!
def runner(
    params,

    demo=True
):

    

    model = ModelCollection[params['model_params']['name']](**params['model_params'])

    training_dataset, validating_dataset, testing_dataset = GrepDataset(**params['data_params'])
    
    batchSize = params['exp_params']['train_batch_size']


    T_loader = DataLoader(
        training_dataset, shuffle=True, batch_size = batchSize
    )
    
    
    V_loader = DataLoader(
        validating_dataset, shuffle=True, batch_size = params['exp_params']['valid_batch_size']
    )


    Te_loader = DataLoader(
        testing_dataset, shuffle=False, batch_size = len(testing_dataset)
    )

    
    EXP = trainer(
        model,
        T_loader,
        V_loader,
        Te_loader,
        params['exp_params'],
        params['logging_params'],
        demo = demo
    )

    EXP.run(echo=False)





def mass_run():    

    for train_amount in [1000, 5000, 100000]:

        valid_amount = int(train_amount / 10)

        test_amount = int(train_amount / 10)

        for data_source in ["MNL", "MC", "NP"]:

            for data_size in [20, 50]:

                data_path = "EXP1_datasets/"+data_source+"/NProd_"+str(data_size)

                ## we use 20 seeds to grep data

                for seed in range(1234, 1235):

                    for net in ["VanillaMNL", "AssortmentNN", "ResAssortNN"]:

                        if net == "VanillaMNL":
                            continue

                        print("Model : ",net)
                        print("Data : "+data_source+"_"+str(data_size))
                        print("Data Amount : "+str(train_amount))



                        yaml_path = "configs/EXP1_"+net+".yaml"

                        with open(yaml_path, 'r') as file:

                            params = yaml.safe_load(file)

                        ## modification


                        params["model_params"]["Nprod_Veclen"] = data_size
                        params["data_params"]["data_seed"] = seed

                        params["data_params"]["data_path"] = data_path

                        params["data_params"]["train_amount"] = train_amount

                        params["data_params"]["valid_amount"] = valid_amount
  
                        params["data_params"]["test_amount"] = test_amount


                        params["exp_params"]["exp_seed"] = seed
 
                        runner(params)












