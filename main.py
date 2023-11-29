from mapa import criar
import pandas as pd
from time import time as t
from funcoes_base import nota
import cProfile
from numpy.polynomial.legendre import leggauss
import pstats
import numpy as np
from scipy.stats import norm

def irt_3pl_model(a, b, c, theta):
    """Calculate the probability of a correct response on an item."""
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))

def eap_estimation(item_params, responses):
    """Estimate ability using EAP method."""
    theta_values = np.linspace(-2, 2, 1000)  # Range of theta values
    likelihoods = np.ones_like(theta_values)
    
    for response, (a, b, c) in zip(responses, item_params):
        p_correct = irt_3pl_model(a, b, c, theta_values)
        likelihoods *= p_correct if response == 1 else (1 - p_correct)
    
    posterior = likelihoods * norm.pdf(theta_values)  # Prior is standard normal
    posterior /= np.sum(posterior)
    
    return np.sum(posterior * theta_values)



def main():
    file_path = 'guarda.csv'
    df = pd.read_csv(file_path, sep=';')
    selected_columns = df[['NU_PARAM_A', 'NU_PARAM_B', 'NU_PARAM_C']]
    Csi = selected_columns.values.tolist()
    # Csi = [sublist for sublist in Csi if not any(pd.isna(value) for value in sublist)]
    Csi = Csi[::-1]
    # print(len(Csi))
    # print(len(sorted(Csi, key=lambda x: x[1])))
    # criar(3, sorted(Csi, key=lambda x: x[1]), nquest = len(Csi))
    Xq, A = leggauss(40)
    # Xq = [i  for i in Xq]
    # estimated_ability = eap_estimation(Csi, [1,0,0,0,0,0,0,1,0,0,1,1,1,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,1,0,0,1,1,0,1,0])
    # print("Estimated Ability:", estimated_ability)
    print(nota([0,1,0,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,1,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1],Csi,Xq,A,41))
    # print(nota([1,0,0,0,0,0,0,1,0,0,1,1,1,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,1,0,0,1,1,0,1,0],Csi,Xq,A,43))
    # print(nota([0 if not i in [41,42,43] else 1 for i in range(43)],Csi,Xq,A,43))
if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    s1 = t()
    main()
    print(t()-s1)
    # print(len([1,0,0,0,0,0,0,1,0,0,1,1,1,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,1,0,0,1,1,0,1,0]))
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(10)