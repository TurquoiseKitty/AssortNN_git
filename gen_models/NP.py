import numpy as np
import os


class NP:

    def __init__(

        self, 
        N_prod = 10, 

        N_mix = 5,

        **kwargs

    ):

        self.ModelName = "NP"

        self.N_prod = N_prod

        self.N_mix = N_mix

        self.mixture_para = np.zeros(N_mix)
        self.permutate_para = np.zeros((N_mix, N_prod))

        self.self_gen_instance()


    def self_gen_instance(self):

        probs = np.random.uniform(low=0., high=1, size=self.N_mix)
        self.mixture_para = probs / sum(probs)

        for i in range(self.N_mix):
            self.permutate_para[i] = np.random.permutation(self.N_prod)


    def prob_for_assortment(self, assortment):

        bundle = np.array([i for i in range(self.N_prod) if assortment[i] == 1 ])
        
        probs = np.zeros(self.N_prod)
        for mix in range(self.N_mix):
            mix_para = self.mixture_para[mix]
            util_paras = self.permutate_para[mix]

            for idx in range(self.N_prod):
                if util_paras[idx] in bundle:
                    
                    probs[int(util_paras[idx])] += mix_para
                    break

        return probs

    
    def gen_final_choice(self, assortment):


        cati = np.random.choice(self.N_mix, 1, p=self.mixture_para)[0]

        permu = self.permutate_para[cati]
        
        ret = np.zeros(self.N_prod)

        for idx in range(self.N_prod):
            if assortment[int(permu[idx])] == 1:
                ret[int(permu[idx])] = 1
                break

        return ret



    def save_para(self, folder_path):

        path = folder_path

        
        with open(folder_path+"/TrueModel_description.txt", 'w') as f:
            f.write("model type : "+self.ModelName)
            f.write("\n")
            f.write("product number : "+str(self.N_prod))
            f.write("\n")
            f.write("mixture number : "+str(self.N_mix))
            f.write("\n")
            

        np.save(folder_path+"/permutate_para", self.permutate_para)































