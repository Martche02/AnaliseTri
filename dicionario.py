import pandas as pd
from time import time
from multiprocessing import Pool
from tempo import carQ
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

def calcularGanho(A, L, nota):
    return A*nota+L
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
def count_bits(n):
    """
    Brian Kernighan's Algorithm
    """
    count = 0
    while n:
        n &= (n - 1)  # Desliga o bit '1' menos significativo
        count += 1
    return count
def diferenca(a,b):
    r ='0'
    d = '0'*(45-len(bin(b)[2:]))+bin(b)[2:]
    for idx, n in enumerate(bin(a)[2:]):
        r+= n if n==d[idx] else '0' if n==1 else '2'
    
    return int(r,2) if '2' not in r else 0

    pass
def find_first_power_of_two_diff(df, serial_number, batch_size=1000):
    """
    Encontra o primeiro número serial no DataFrame e sua nota cuja diferença com o 
    número serial fornecido é uma potência de 2, processando em lotes.
    Retorna apenas o primeiro número serial encontrado.
    """
    # df = df.sample(2000)
    df["dif"] = df['Acertos_Decimal'].apply(lambda x: count_bits(x^serial_number if x!=serial_number else 2**45-1))# if x!=serial_number else 0)
    indice = df['dif'].idxmin()
    print(df["dif"][indice])
    # print(df.sort_values(by="dif", ascending=True)["dif"].head(5))
    # print(df.sort_values(by="dif", ascending=True)[1])
    #'000000000000010000100000011011100101000000111'
    #'000000000000010000100000011011100101000000111'
    return df["Acertos_Decimal"][indice], df[df.columns[1]][indice]
    num_batches = int(np.ceil(len(df) / batch_size))

    for batch in range(num_batches):
        start_index = batch * batch_size
        end_index = start_index + batch_size
        batch_df = df.iloc[start_index:end_index]

        serial_numbers = batch_df['Acertos_Decimal'].to_numpy()
        diffs = np.abs(serial_numbers - serial_number)

        is_power_of_two = (diffs & (diffs - 1) == 0) & (diffs != 0)
        indices = np.where(is_power_of_two)[0]
        
        if len(indices) > 0:
            print(serial_numbers[indices[0]], batch_df[indices[0]][batch_df.columns[1]])
            return serial_numbers[indices[0]], batch_df[indices[0]][batch_df.columns[1]]

    return None, None
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
    df1 = pd.read_csv(f"{ANO}dados/{cod}.0.csv") # big data
    df2 = pd.read_csv(f"{ANO}dadositens/{cod}.csv") #apenas coeficientes
    Angular_C, Linear_C = df2["Angular_C"].values.tolist(), df2["Linear_C"].values.tolist()
    
    # Presumo que as colunas Angular_C e Linear_C não são utilizadas no seu código original.
    # tirar = serial
    # df_filtrado = df1[df1['Acertos_Decimal'] != tirar]
    # valores = df_filtrado['Acertos_Decimal'].values

    # Filtra os valores menores que o serial e encontra o maior deles
    # valores_menores_que_serial = valores[valores < tirar]
    # if valores_menores_que_serial.size == 0:
    #     return None  # Retorna None se não houver nenhum valor menor que serial
    # valor_max_menor_que_serial = np.max(valores_menores_que_serial)

    # # Encontra o índice do valor_max_menor_que_serial
    # indice_max_menor_que_serial = np.where(valores == valor_max_menor_que_serial)[0][0]
    
    # linha_mais_proxima = df_filtrado.iloc[indice_max_menor_que_serial]
    # tirar = valor_max_menor_que_serial
    # nota_d = linha_mais_proxima["NU_NOTA_CN"]
    serial_e, nota_d = find_first_power_of_two_diff(df1, serial)
    # diferenca = abs(serial - valor_max_menor_que_serial)
    # print(bin(diferenca))
    nota = nota_d
    serial = '0'*(45-len(bin(serial)[2:]))+bin(serial)[2:]
    serial_e = '0'*(45-len(bin(serial_e)[2:]))+bin(serial_e)[2:]
    variacao = 0
    for i in range(45):
        variacao +=(int(serial_e[i])-int(serial[i]))*(Angular_C[i]*nota_d+Linear_C[i]) 
        nota+=(int(serial_e[i])-int(serial[i]))*(Angular_C[i]*nota_d+Linear_C[i]) 
    print(variacao)
    # for idx, n in enumerate('0'*(45-len(bin(diferenca)[2:]))+bin(diferenca)[2:]):
    #     if n=='1':
    #         nota+=(Angular_C[idx]*nota_d+Linear_C[idx]) #/(Angular_C[idx]+1)
    # print(bin(diferenca)[2:].count('1'))
    # print(diferenca)
    return nota # if bin(diferenca)[2:].count('1') == 1 else -sum(notas)/len(notas)
    # r = sum(notas)/len(notas)
    # return r if bin(diferenca)[2:]

def mostrar_acuracia():
    df = pd.read_csv('2022dados/1087.0.csv')
    df = df.sample(n=20)  # Seleciona aleatoriamente 100 linhas
    # df['Mapeada'] = df['Acertos_Decimal'].apply(mapear_resp)
    # df['Approx_Nota'] = df['Mapeada'].apply(notaProx)
    df['NotaAprox'] = df["Acertos_Decimal"].apply(notaProx)

    # Selecting a sample of 2000 points for plotting to avoid overplotting
    #plot_sample = df #.sample(n=2000, random_state=42)
    
    # Calculate the difference array
    difference_array = df['NotaAprox'] - df['NU_NOTA_CN']
    # difference_array = difference_array[difference_array>-200]
    # print(difference_array)
    # difar = df['Approx_Nota'] - df['NU_NOTA_CN']

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.scatter(df['NU_NOTA_CN'], difference_array, alpha=0.5, label='Approximated vs Actual Nota')
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
        'std_dev': np.std(difference_array),
        'variance': np.var(difference_array)
    }
    print(statistics)

def achar_melhor_opcao(serial):
    nota = notaProx(serial)
    vet = (45-len(bin(serial)[2:]))*'0'+bin(serial)[2:]
    df = pd.read_csv('2022dadositens/1087.csv')
    Angular_C, Linear_C = df["Angular_C"], df["Linear_C"]
    opcoes = [calcularGanho(Angular_C[i],Linear_C[i],nota)*int(vet[i]) for i in range(45)]
    print("Questao "+str(opcoes.index(max(opcoes))+90), max(opcoes))

def mostrar_graficos():
    for i in range(45):
        carQ(pd.read_csv(f"2022dados/1087.0.csv"), i, True)










if __name__ == '__main__':
    t = time()

    # achar_melhor_opcao(1585143219722) # serial
    mostrar_acuracia()
    # print(notaProx(13220013949184))
    # mostrar_graficos()

    print(f"time: {time()-t} seg")