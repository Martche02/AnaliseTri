import pandas as pd
from time import time
from multiprocessing import Pool
from tempo import carQ
import numpy as np
import os
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
def lerc(path:str, *args:str)->pd.DataFrame:
    return pd.read_csv(path, usecols=args, header=0, delimiter=';', engine="pyarrow")
def criarTabGab(ANO:int|str)->None:
    l = ["CO_POSICAO","SG_AREA", "CO_ITEM", "TX_GABARITO", "NU_PARAM_A", "NU_PARAM_B", "NU_PARAM_C", "CO_PROVA"]
    df = lerc(str(ANO)+"/DADOS/ITENS_PROVA_"+str(ANO)+".csv", *l)
    for i in df["CO_PROVA"].dropna().unique().tolist():
        df[df["CO_PROVA"]==i].to_csv(str(ANO)+"dadositens/"+str(i)+".csv", index=False)
def criarTabRes(ANO:int|str)->None:
    c, d = ["TX_RESPOSTAS_", "NU_NOTA_", "CO_PROVA_"], ["LC", "CH", "CN", "MT"]
    l = [a+b for a in c for b in d]
    df = lerc(str(ANO)+"/DADOS/MICRODADOS_ENEM_"+str(ANO)+".csv", *l)
    gab = dict()
    for m in d:
        for i in df[c[2]+m].dropna().unique().tolist():
            df[df[c[2]+m]==i][[j+m for j in c[:2]]].to_csv(str(ANO)+"dados/"+str(i)+".csv", index=False)
            gab[i] = lerc(str(ANO)+"/DADOS/MICRODADOS_ENEM_"+str(ANO)+".csv", *[j for j in [c[2]+m, "TX_GABARITO_"+m]]).set_index(c[2]+m).loc[i,"TX_GABARITO_"+m].iloc[0]
    pd.DataFrame(list(gab.items()), columns=['codigo', 'gabarito']).to_csv(str(ANO)+"dados/gabarito.csv", index=False)
def find_matching_responses_and_check_uniformity(my_response, answer_key, df, column_index=1):
    if len(my_response) != 45 or len(answer_key) != 45:
        raise ValueError("Responses and answer key must be 45 characters long.")
    pattern = ''
    for my_ans, correct_ans in zip(my_response, answer_key):
        if my_ans == correct_ans or my_ans == '1':
            pattern += correct_ans
        else:
            pattern += f'[^{correct_ans}]'
    matching_column_data = df[df[df.columns[0]].str.match(pattern)].iloc[:, column_index]
    # print(pattern, answer_key, my_response)
    if not matching_column_data.empty:
        return matching_column_data.iloc[0] if matching_column_data.nunique() == 1 else "Variance in responses:"+str(matching_column_data)
def nota_por_prova(ANO:int|str, CO_PROVA:int|str, respostas:str)->float|None:
    if respostas != 'X'*len(respostas):
        codigo = str(CO_PROVA)
        df = pd.read_csv(str(ANO)+"dados/"+codigo+".0.csv")
        gabarito = ''.join(pd.read_csv(str(ANO)+"dadositens/"+codigo+".csv").sort_values(by='CO_POSICAO')["TX_GABARITO"].tolist())
        return find_matching_responses_and_check_uniformity(respostas, gabarito, df)
def mapear_resp(serial:str, ANO:int=2022, cod:str|int=1087, cod_i:str|int=1085):
    respostas = '0'*(45-len(bin(serial)[2:]))+bin(serial)[2:]
    original = pd.read_csv(str(ANO)+"dadositens/"+str(cod)+".csv").sort_values(by='CO_POSICAO')["CO_ITEM"].tolist()
    nova = pd.read_csv(str(ANO)+"dadositens/"+str(cod_i)+".csv").sort_values(by='CO_POSICAO')["CO_ITEM"].tolist()
    # print(original)
    # print(nova)
    a = original
    b = nova
    # print(set(original)==set(nova)) # it was true
    respostas_mapeadas = ''.join([dict(zip(b,respostas)).get(questao, "X") for questao in a])
    return int(respostas_mapeadas,2)
def nota(ANO:int, prova:str, cod:str|int, respostas:str)->float|None:
    """ANO = 2022 | 2021 | 2020 etc.\n
    PROVA = LC | CH | CN | MT\n
    COD = 1999\n
    RESPOSTAS = 'abcdeabcdeabcdeabcde...abcde'"""
    original = pd.read_csv(str(ANO)+"dadositens/"+str(cod)+".csv").sort_values(by='CO_POSICAO')["CO_ITEM"].tolist()
    for i in os.listdir(str(ANO)+"dados/"):
        df = pd.read_csv(os.path.join(str(ANO)+"dados/", i))
        if df.columns[0][-2:] == prova:
            nova = pd.read_csv(str(ANO)+"dadositens/"+i[:4]+".csv").sort_values(by='CO_POSICAO')["CO_ITEM"].tolist()
            respostas_mapeadas = ''.join([dict(zip(original, respostas)).get(questao, "X") for questao in nova])
            n = nota_por_prova(ANO,i[:4],respostas_mapeadas)
            if n is not None:
                return n
def process_file(args):
    ANO, prova, cod, respostas, filename = args
    original = pd.read_csv(str(ANO)+"dadositens/"+str(cod)+".csv").sort_values(by='CO_POSICAO')["CO_ITEM"].tolist()
    df = pd.read_csv(os.path.join(str(ANO)+"dados/", filename))
    if df.columns[0][-2:] == prova:
        nova = pd.read_csv(str(ANO)+"dadositens/"+filename[:4]+".csv").sort_values(by='CO_POSICAO')["CO_ITEM"].tolist()
        respostas_mapeadas = ''.join([dict(zip(original, respostas)).get(questao, "X") for questao in nova])
        n = nota_por_prova(ANO, filename[:4], respostas_mapeadas)
        if n is not None:
            return n
    return None
def nota_parallel(ANO: int, prova: str, cod: str|int, respostas: str) -> float|None:
    """ANO = 2022 | 2021 | 2020 etc.\n
    PROVA = LC | CH | CN | MT\n
    COD = 1999\n
    RESPOSTAS = 'abcdeabcdeabcdeabcde...abcde'"""
    files = os.listdir(str(ANO)+"dados/")
    with Pool() as pool:
        # Use imap or imap_unordered to get results as they are ready
        for result in pool.imap_unordered(process_file, [(ANO, prova, cod, respostas, f) for f in files]):
            if result is not None:
                # Terminate other processes and return the result
                pool.terminate()
                return result
    return None
def binarizar(ANO):
    cods = os.listdir(str(ANO)+"dados/")
    for cod in cods:
        df = pd.read_csv(str(ANO)+"dados/"+str(cod))
        if df.columns[0][-2:] != 'LC' and str(cod) !='gabarito.csv':
            t = time()
            gabarito = ''.join(pd.read_csv(str(ANO)+"dadositens/"+str(cod)[:-6]+".csv").sort_values(by='CO_POSICAO')["TX_GABARITO"].tolist())
            gabarito_array = np.array(list(gabarito))
            respostas_array = np.array(df[df.columns[0]].apply(list).tolist())
            acertos = (respostas_array == gabarito_array).astype(int)
            df['Acertos_Decimal'] = [int(''.join(map(str, acerto)), 2) for acerto in acertos]
            df.to_csv(str(ANO)+"dados/"+str(cod), index=False)
            print(f"time: {time()-t} seg")
def addLine(ANO:str|int, cod:str|int):
    ANO, cod = str(ANO), str(cod)
    df = pd.read_csv(f"{ANO}dadositens/{cod}.csv")
    df.sort_values(by=df.columns[0], ascending=False, inplace=True)
    external_data = pd.read_csv(f"{ANO}dados/{cod}.0.csv")
    length = len(df)
    a,b=[],[]
    for idx in range(length):
        ai, bi = carQ(external_data, idx)
        print(ai,bi)
        a.append(ai)
        b.append(bi)
    df["Angular_C"] = a
    df["Linear_C"] = b
    df.to_csv(f"{ANO}dadositens/{cod}.csv", index=False)
def aproxNota(serial:int, ANO:str|int=2022, cod:str|int=1087)->float: #ta dando muito pra cima = nota máxima. Se conseguir dar nota mínima fechou todas
    ANO, cod = str(ANO), str(cod)
    df = pd.read_csv(f"{ANO}dadositens/{cod}.csv")
    inf = 360.6 #pd.read_csv(f"{ANO}dados/{cod}.0.csv")
    Angular_C, Linear_C = df["Angular_C"].values.tolist(), df["Linear_C"].values.tolist()
    num, den = 0, 0
    for idx, n in enumerate('0'*(45-len(bin(serial)[2:]))+bin(serial)[2:]):
        num+=int(n)*float(Linear_C[idx])/(float(Angular_C[idx])+1)
        den+=int(n)*float(Angular_C[idx])/(float(Angular_C[idx])+1)
    return (num+inf)/(1-den)
def notaProx(serial:int, ANO:str|int=2022, cod:str|int=1087)->float:
    ANO, cod = str(ANO), str(cod)
    df1 = pd.read_csv(f"{ANO}dados/{cod}.0.csv")
    df2 = pd.read_csv(f"{ANO}dadositens/{cod}.csv")
    Angular_C, Linear_C = df2["Angular_C"].values.tolist(), df2["Linear_C"].values.tolist()
    df_filtrado = df1[df1['Acertos_Decimal'] != serial]
    diferencas_abs = np.abs(df1['Acertos_Decimal'].values - serial)
    indice_mais_proximo = np.argmin(diferencas_abs)
    linha_mais_proxima = df1.iloc[indice_mais_proximo]
    nota_d = linha_mais_proxima["NU_NOTA_CN"]
    valor_nota_cn_mais_proximo = df_filtrado.iloc[indice_mais_proximo]['Acertos_Decimal']
    diferenca = abs(serial - valor_nota_cn_mais_proximo)
    nota = nota_d
    for idx, n in enumerate('0'*(45-len(bin(diferenca)[2:]))+bin(diferenca)[2:]):
        if n==1:
            nota+=(Angular_C[idx]*nota_d+Linear_C[idx]) #/(Angular_C[idx]+1)
    return nota



if __name__ == '__main__':
    t = time()
    # for i in range(45):
    # carQ(pd.read_csv(f"2022dados/1087.0.csv"), 0, True)
    # df = pd.read_csv(f"2022dadositens/1087.csv")
    # column_to_shift = "Angular_C"
    # # Shift the column
    # df[column_to_shift] = df[column_to_shift].shift(-1)

    # df.iloc[-1, df.columns.get_loc(column_to_shift)] = df[column_to_shift].iloc[0]
    # df.to_csv("2022dadositens/1087.csv")

    # addLine(2022, 1087)
    
    # carQ(pd.read_csv(f"2022dados/1087.0.csv"), 44, True)
    # pd.concat([pd.read_csv("2022dadositens/108586.csv"),pd.read_csv("2022dadositens/1087.csv")], ignore_index=True).to_csv("2022dadositens/10858687.csv")
    # pd.concat([pd.read_csv("2022dados/108586.0.csv"),pd.read_csv("2022dados/1087.0.csv")], ignore_index=True).to_csv("2022dados/10858687.0.csv")
    # addLine(2022,108586)
    # addLine(2022,1087)
    # print(notaProx(118364175,2022,1087))
    # Apply the aproxNota function to the 'Acertos_Decimal' column
    import matplotlib.pyplot as plt
    df = pd.read_csv('2022dados/1085.0.csv')
    df = df.sample(n=200)  # Seleciona aleatoriamente 100 linhas
    df['Mapeada'] = df['Acertos_Decimal'].apply(mapear_resp)
    df['Approx_Nota'] = df['Mapeada'].apply(notaProx)

    # Selecting a sample of 2000 points for plotting to avoid overplotting
    plot_sample = df #.sample(n=2000, random_state=42)
    
    # Calculate the difference array
    difference_array = df['Approx_Nota'] - df['NU_NOTA_CN']

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.scatter(plot_sample['NU_NOTA_CN'], difference_array, alpha=0.5, label='Approximated vs Actual Nota')
    plt.xlabel('NU_NOTA_CN')
    plt.ylabel('Approximated Nota by aproxNota')
    plt.legend()
    plt.title('Scatter Plot of Actual vs. Approximated Notas')
    plt.grid(True)
    plt.show()


    # Calculate statistics
    statistics = {
        'mean': np.mean(difference_array),
        'median': np.median(difference_array),
        # 'mode': stats.mode(difference_array)[0][0],
        'std_dev': np.std(difference_array),
        'variance': np.var(difference_array)
    }
    print(statistics)
    # import matplotlib.pyplot as plt
    # start = 1
    # end = 45

    # # Generate 400 equally spaced values in the range
    # x_values = np.linspace(start, end)

    # # Calculate the output for each value
    # y_values = [aproxNota(int((45-int(x))*'0'+int(x)*'1',2),2022,1087) for x in x_values]
    # y2_values = [aproxNota(int((int(x))*'1'+(45-int(x))*'0',2),2022,1087) for x in x_values]

    # # Plotting
    # plt.figure(figsize=(12, 6))
    # plt.plot(x_values, y_values, y2_values, label='aproxNota(x)')
    # plt.xlabel('Input Value')
    # plt.ylabel('Output Value')
    # plt.title('Function Behavior of aproxNota')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    print(f"time: {time()-t} seg")