import numpy as np
import os


class MNL:

    def __init__(

        self, 
        N_prod = 10, 

        Sigma = 1,

        **kwargs

    ):

        self.ModelName = "MNL"

        self.N_prod = N_prod

        self.utils = np.zeros(N_prod)

        self.self_gen_instance(Sigma)


    def self_gen_instance(self, Sigma):

        self.utils = np.random.normal(0, Sigma, self.N_prod)


    def prob_for_assortment(self, prod_assort):

        probs = np.exp(self.utils) * prod_assort

        probs = probs / sum(probs)

        return probs

    # the output is also one_hot encoded
    def gen_final_choice(self, prod_assort):

        fin = np.zeros(self.N_prod)
        
        probs = self.prob_for_assortment(prod_assort)

        choice = np.random.choice(self.N_prod, 1, p=probs)[0]

        fin[choice] = 1

        return fin


    def save_para(self, folder_path):

        path = folder_path

        
        with open(folder_path+"/TrueModel_description.txt", 'w') as f:
            f.write("model type : "+self.ModelName)
            f.write("\n")
            f.write("product number : "+str(self.N_prod))
            f.write("\n")
            

        np.save(folder_path+"/utils", self.utils)




if __name__ == "__main__":

    mnl = MNL(
        N_prod = 20
    )

    print(mnl.utils)
    assortment = np.array([1] * 20)

    print(mnl.prob_for_assortment(assortment))



