from mapa import criar
import pandas as pd
from time import time as t
import cProfile
import pstats

def main():
    file_path = 'guarda.csv'
    df = pd.read_csv(file_path, sep=';')
    selected_columns = df[['NU_PARAM_A', 'NU_PARAM_B', 'NU_PARAM_C']]
    Csi = selected_columns.values.tolist()
    Csi = [sublist for sublist in Csi if not any(pd.isna(value) for value in sublist)]
    # print(len(sorted(Csi, key=lambda x: x[1])))
    criar(5, sorted(Csi, key=lambda x: x[1]), nquest = len(Csi))
if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    s1 = t()
    main()
    print(t()-s1)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(10)