import numpy as np
import random

# for given 0 < lam < 1, given N candidate products, generated assortment will contain averagely lam * N products
def GenAssortment_Even(N_prod = 10, lam=1/2, **kwargs):
    potential_vec = np.random.uniform(low=0., high=1, size=N_prod)
    assortment_vec = np.zeros(N_prod)
    assortment_vec[potential_vec <= lam] = 1
    return assortment_vec


# generate assortment containing fixed number of products
def GenAssortment_Fixed(N_prod = 10, fixed_num = 6, **kwargs):
    positions = random.sample(list(range(N_prod)),k=fixed_num)
    assortment_vec = np.zeros(N_prod)
    assortment_vec[positions] = 1
    return assortment_vec

def GenAssortment_Abundant(N_prod = 10, **kwargs):
    fixied = random.sample(list(range(1,N_prod+1)),k=1)[0]
    return GenAssortment_Fixed(N_prod, fixied)



GenAssortment = {
    "Even" : GenAssortment_Even,
    "Fixed": GenAssortment_Fixed,
    "Abundant": GenAssortment_Abundant
}
