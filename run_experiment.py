from train import trainer, KL_loss
import yaml
from models import ModelCollection
from dataset import GrepDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

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

    return EXP.model


all_configs = [
    ## train_amount, data_source, data_size
    (5000, "MC", 20),
    (5000, "NP", 20),
    (5000, "NP", 50),

    (100000, "MC", 20),
    (100000, "NP", 50)

]




def mass_run(demo = True):   

    for train_amount, data_source, data_size in all_configs: 

    

        valid_amount = int(train_amount / 10)

        test_amount = int(train_amount / 10)


        data_path = "EXP1_datasets/"+data_source+"/NProd_"+str(data_size)

        ## we use 20 seeds to grep data

        harvestor = {
            "VanillaMNL" : np.array([]),
            "VanillaMNL_best" : np.array([]),
            "AssortmentNN" : np.array([]),
            "AssortmentNN_best" : np.array([]),
            "ResAssortNN" : np.array([]),
            "ResAssortNN_best" : np.array([]),
        }

        for seed in range(1234, 1254):

            for net in ["VanillaMNL", "AssortmentNN", "ResAssortNN"]:

                if net == "VanillaMNL" and train_amount == 100000:

                    continue

                if demo:
                    print("Model : ",net)
                    print("Data : "+data_source+"_"+str(data_size))
                    print("Data Amount : "+str(train_amount))



                yaml_path = "configs/EXP1_"+net+".yaml"

                with open(yaml_path, 'r') as file:

                    params = yaml.safe_load(file)

                ## modification
                if train_amount == 100000:

                    params["exp_params"]["max_epochs"] = 50


                params["model_params"]["Nprod_Veclen"] = data_size
                params["data_params"]["data_seed"] = seed

                params["data_params"]["data_path"] = data_path

                params["data_params"]["train_amount"] = train_amount

                params["data_params"]["valid_amount"] = valid_amount

                params["data_params"]["test_amount"] = test_amount


                params["exp_params"]["exp_seed"] = seed

                model = runner(params, demo = demo)

                log_path = "logs/"+params["logging_params"]["log_name"]

                ## leave the validation plot

                train_loss = np.load(log_path+"/log_train_loss.npy")
                valid_loss = np.load(log_path+"/log_valid_loss.npy")

                if demo: 

                    plot_name = net+"_on_"+data_source+str(data_size)+"_withAmount_"+str(train_amount)


                    plt.clf()
                    plt.plot(train_loss)
                    plt.plot(valid_loss)
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'val'], loc='upper right')

                    plt.savefig("plots/"+plot_name+".png")

                test_loss, best_loss = KL_loss_check(params, model)

                harvestor[net] = np.append(harvestor[net], test_loss)
                harvestor[net+"_best"] = np.append(harvestor[net+"_best"], best_loss)

        ## store harvestor
        harvestor_name = data_source+str(data_size)+"_withAmount_"+str(train_amount)
        np.save("harvestors/"+harvestor_name+".npy", harvestor)





def KL_loss_check(params, model):

    training_dataset, validating_dataset, testing_dataset = GrepDataset(**params['data_params'])

    model.eval()

    IN, SAMP, PROB = testing_dataset[:]

    model_OUT = model(IN)

    best_OUT = PROB

    test_loss = KL_loss(model_OUT, SAMP)['KL_loss'].detach().cpu().item()
    best_loss = KL_loss(best_OUT, SAMP)['KL_loss'].detach().cpu().item()

    return (test_loss, best_loss)


if __name__ == "__main__":

    mass_run(demo= False)












