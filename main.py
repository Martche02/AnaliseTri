from mapa import criar
import pandas as pd
import time
start = time.time()
file_path = 'guarda.csv'
df = pd.read_csv(file_path, sep=';')
selected_columns = df[['NU_PARAM_A', 'NU_PARAM_B', 'NU_PARAM_C']]
Csi = selected_columns.values.tolist()
Csi = [sublist for sublist in Csi if not any(pd.isna(value) for value in sublist)]
# print(len(sorted(Csi, key=lambda x: x[1])))
criar(3, sorted(Csi, key=lambda x: x[1]), nquest = len(Csi))
print(time.time()-start)