from multiprocessing import Pool
import matplotlib.pyplot as plt
from time import time
import pandas as pd
import numpy as np
import sqlite3
import csv
import os

iab = False
if iab:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from tensorflow.keras.models import load_model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

def ia(ANO, cod):
    df = pd.read_csv(f"C:/Users/Marce/Codes/Frezza Fisica/"+str(ANO)+"dados/"+str(cod)+".0.csv")
    q = 45
    df[df.columns[1]] = df[df.columns[1]] / 1000 
    df['Acertos_Binario'] = df['Acertos_Decimal'].apply(lambda x: format(x, '045b')) 
    for i in range(q):
        df[f'bit_{i}'] = df['Acertos_Binario'].apply(lambda x: int(x[i]))
    X = df[[f'bit_{i}' for i in range(q)]]
    y = df[df.columns[1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential([
        Dense(q, activation='relu', input_shape=(q,)),
        Dense(q, activation='relu'),
        Dense(1, activation='sigmoid')  # Saída entre 0 e 1
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.fit(X_train, y_train, epochs=20, validation_split=0.1)
    loss = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}')
    predictions = model.predict(X_test)  # Desnormalizar as previsões
    errors = predictions.flatten() - y_test  # Calcula a diferença entre o previsto e o real
    print("Desvio padrão dos erros:", np.std(errors))
    model_directory = f"C:/Users/Marce/Codes/Frezza Fisica/models/{ANO}_{cod}"
    model.save(model_directory)
def dec2vet(serial):
    return format(int(serial), '045b')
def find_pairs_with_difference(df, X):
    # Ordenar o DataFrame pelos números seriais
    sorted_df = df.sort_values(by='Acertos_Decimal')

    # Converter para array NumPy para acesso mais rápido
    serial_numbers = sorted_df['Acertos_Decimal'].to_numpy()
    notes = sorted_df[df.columns[1]].to_numpy()
    
    pairs = []
    for i in range(len(serial_numbers)):
        # Busca binária por um par
        target = serial_numbers[i] + X
        j = np.searchsorted(serial_numbers, target, side='left')
        if j < len(serial_numbers) and serial_numbers[j] == target:
            pairs.append((notes[i], notes[j]))
    return pairs
def find_pairs_with_differencedif(df, X):
    # Ordenar o DataFrame pelos números seriais
    sorted_df = df.sort_values(by='Acertos_Decimal')

    # Converter para array NumPy para acesso mais rápido
    serial_numbers = sorted_df['Acertos_Decimal'].to_numpy()
    notes = sorted_df[df.columns[1]].to_numpy()
    
    pairs = []
    for i in range(len(serial_numbers)):
        # Busca binária por um par
        target = serial_numbers[i] + X
        j = np.searchsorted(serial_numbers, target, side='left')
        if j < len(serial_numbers) and serial_numbers[j] == target:
            pairs.append((serial_numbers[i], serial_numbers[j]))
    return pairs
def carQ(df:pd.DataFrame, X:int)->float:
    X = 2**X
    l = set(find_pairs_with_difference(df, X))
    x_vals = np.array([])
    y_vals = np.array([])
    try:
        l = [pair for pair in l if pair[0] != 0 and pair[1]>1]
        for pair in l:
            lower, higher = min(pair), max(pair)
            x_vals = np.append(x_vals,[lower])
            y_vals = np.append(y_vals,[higher-lower])
        lin_reg = LinearRegression()
        lin_reg.fit(x_vals.reshape(-1, 1), y_vals)
        predicted_y_vals = lin_reg.predict(x_vals.reshape(-1, 1))
        above_line_mask = y_vals > predicted_y_vals
        x_vals = x_vals[above_line_mask]
        y_vals = y_vals[above_line_mask]
        if y_vals.size > 0:
            return sum(y_vals)/len(y_vals)
        else:
            print(y_vals)
            return 0
    except ValueError:
        return 0
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
    print(cod)
    ANO, cod = str(ANO), str(cod)
    df = pd.read_csv(f"{ANO}dadositens/{cod}.csv")
    df.sort_values(by=df.columns[0], ascending=False, inplace=True)
    external_data = pd.read_csv(f"{ANO}dados/{cod}.0.csv")
    length = len(df)
    a=[]
    for idx in range(length):
        ai = float(carQ(external_data, idx))
        print(ai)
        a.append(ai)
    df["Mean"] = a
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
def find_first_power_of_two_diff(df, serial_number):
    df["dif"] = df['Acertos_Decimal'].apply(lambda x: count_bits(x^serial_number if x!=serial_number else 2**45-1))# if x!=serial_number else 0)
    indice = df['dif'].idxmin()
    return df["Acertos_Decimal"][indice], df[df.columns[1]][indice]
def aproxNota(serial:int, ANO:str|int=2022, cod:str|int=1087, min=0)->float: 
    df = pd.read_csv(str(ANO)+"dadositens/"+str(cod)+".csv")
    Mean = df["Mean"].tolist()
    num = 0
    for idx, n in enumerate('0'*(45-len(bin(serial)[2:]))+bin(serial)[2:]):
        num += int(n)*Mean[idx]
    return num+min
def notaProx(serial:int, ANO:str|int=2022, cod:str|int=1087)->float:
    ANO, cod = str(ANO), str(cod)
    df1 = pd.read_csv(f"{ANO}dados/{cod}.0.csv") # big data
    df2 = pd.read_csv(f"{ANO}dadositens/{cod}.csv") #apenas coeficientes
    Angular_C, Linear_C = df2["Angular_C"].values.tolist(), df2["Linear_C"].values.tolist()
    df1["dif"] = df1['Acertos_Decimal'].apply(lambda x: count_bits(x^serial if x!=serial else 2**45-1))
    indice = df1['dif'].idxmin()
    serial_e, nota_d = df1["Acertos_Decimal"][indice], df1[df1.columns[1]][indice]
    nota = nota_d
    serial = '0'*(45-len(bin(serial)[2:]))+bin(serial)[2:]
    serial_e = '0'*(45-len(bin(serial_e)[2:]))+bin(serial_e)[2:]
    variacao = 0
    for i in range(45):
        variacao +=(int(serial_e[i])-int(serial[i]))*(Angular_C[i]*nota_d+Linear_C[i]) 
        nota+=(int(serial_e[i])-int(serial[i]))*(Angular_C[i]*nota_d+Linear_C[i]) 
    return nota
def mostrar_acuracia(n, ANO, cod, cod_i, p:bool=False, min=0):
    df = pd.read_csv(str(ANO)+"dados/"+str(cod_i)+'.0.csv')
    # min = df[df.columns[1]].min()
    # min = 0
    df = df.sample(n)  # Seleciona aleatoriamente 100 linhas
    df['Mapeada'] = df['Acertos_Decimal'].apply(lambda x: mapear_resp(x,ANO,cod, cod_i))
    # df['Approx_Nota'] = df['Mapeada'].apply(notaProx)
    df['NotaAprox'] = df["Mapeada"].apply(lambda x: aproxNota(x, ANO, cod, min))

    # Selecting a sample of 2000 points for plotting to avoid overplotting
    #plot_sample = df #.sample(n=2000, random_state=42)
    
    # Calculate the difference array
    # coefficients =  [-8.15704907e-11,  1.96054015e-07, -1.72525142e-04,  6.69846920e-02, -1.03949542e+01,  3.09788668e+02]
    # # coefficients = np.polyfit(df['NU_NOTA_CN'], difference_array, 5)
    # p = np.poly1d(coefficients)
    # difference_array = p(df['NotaAprox'])
    difference_array = df['NotaAprox'] - df[df.columns[1]]
    
    # print(coefficients)
    # difference_array = difference_array[difference_array>-200]
    # print(difference_array)
    # difar = df['Approx_Nota'] - df['NU_NOTA_CN']
    # difference_array = p(df['NotaAprox'])
    # Plot the graph
    if p:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[df.columns[1]], difference_array, alpha=0.5, label='Approximated vs Actual Nota')
        plt.xlabel(str(df.columns[1]))
        plt.ylabel('Error of approximated Nota by aproxNota')
        plt.legend()
        plt.title('Scatter Plot of Actual vs. Approximated Notas')
        plt.grid(True)
        plt.show()


    # Calculate statistics
    statistics = {
        'minimo' : np.mean(difference_array),
        'mean': np.mean(difference_array),
        'median': np.median(difference_array),
        'std_dev': np.std(difference_array),
        'variance': np.var(difference_array)
    }
    print(statistics)
def achar_melhor_opcao(serial, ANO, cod,f):
    # nota = notaProx(serial)
    # vet = (45-len(bin(serial)[2:]))*'0'+bin(serial)[2:]
    vet = dec2vet(serial)
    # df = pd.read_csv('2022dadositens/1087.csv')
    # Angular_C, Linear_C = df["Angular_C"], df["Linear_C"]
    # opcoes = [(i,calcularGanho(Angular_C[i],Linear_C[i],nota)*( 1-int(vet[i]))) for i in range(45)]
    notadoaluno = f(serial, ANO, cod)
    opcoes = [(i+1,f(int(vet[:i] + str(1-int(vet[i])) + vet[i + 1:],2),ANO, cod)*( 1-int(vet[i])) ) for i in range(45)]
    print("Nota:"+str(notadoaluno))
    for i in opcoes:
        if i[1] !=0:
            print("Questao "+str(i[0])+' '+str(i[1]-notadoaluno))
    return max(opcoes, key=lambda x: x[1])
    # print("Questao "+str(opcoes.index(max(opcoes))+90), max(opcoes))
def achar_melhor_opcaoia(serial, ANO, cod):
    # nota = notaProx(serial)
    # vet = (45-len(bin(serial)[2:]))*'0'+bin(serial)[2:]
    vet = dec2vet(serial)
    # df = pd.read_csv('2022dadositens/1087.csv')
    # Angular_C, Linear_C = df["Angular_C"], df["Linear_C"]
    # opcoes = [(i,calcularGanho(Angular_C[i],Linear_C[i],nota)*( 1-int(vet[i]))) for i in range(45)]
    # prever_nota_com_ialote
    notadoaluno = prever_nota_com_ia(serial, ANO, cod)
    lote = [int(vet[:i] + str(1-int(vet[i])) + vet[i + 1:],2) for i in range(45)]
    prever_nota_com_ialote(lote, ANO, cod)
    opcoes = [(i+1,lote[i]*( 1-int(vet[i])) ) for i in range(45)]
    print("Nota:"+str(notadoaluno))
    # for i in opcoes:
    #     if i[1] !=0:
    #         print("Questao "+str(i[0])+' '+str(i[1]-notadoaluno))
    return max(opcoes, key=lambda x: x[1])
    # print("Questao "+str(opcoes.index(max(opcoes))+90), max(opcoes))
def mostrar_graficos(ANO, cod):
    for i in range(45):
        carQ(pd.read_csv(f"{str(ANO)}dados/{str(cod)}.0.csv"), i, True)
def gini(values):
    n = len(values)
    if n == 0:
        return None

    sorted_values = sorted(values)
    sum_values = sum(sorted_values)
    cumulative_values_sum = 0

    for i, value in enumerate(sorted_values, 1):
        cumulative_values_sum += i * value

    gini_index = 1 - (2 / n) * (cumulative_values_sum / sum_values) + (n + 1) / n
    return gini_index
def mk_dir(directory):
    """Cria um diretório se ele não existir."""
    if not os.path.exists(directory):
        os.makedirs(directory)
def calcular_pesos_ANO(ANO:int|str):
    mk_dir(str(ANO) + "dados/")
    mk_dir(str(ANO) + "dadositens/")
    criarTabRes(ANO) #cria os dados
    criarTabGab(ANO) #cria os dadositens
    binarizar(ANO) # cria decimais
    for cod in os.listdir(str(ANO) + "dados/"):
        caminho_arquivo = os.path.join(str(ANO) + "dados/", cod)
        
        # Verifica se o arquivo é maior que 20MB
        if os.path.isfile(caminho_arquivo) and os.path.getsize(caminho_arquivo) > 20 * 1024 * 1024:
            df = pd.read_csv(caminho_arquivo)

            if df.columns[0][-2:] != "LC" and cod != 'gabarito.csv':
                print(df.columns[0][-2:])
                addLine(ANO, str(cod[:-6]))  # calcula pesos
def getMeanValidItens(ANO, cod):
    """Retorna o array com pesos, valor mínimo e gabarito"""
    df = pd.read_csv(str(ANO)+"dadositens/"+str(cod)+".csv")
    m = {'2015': 286, '2016': 327, '2017': 329, '2018':316, '2019': 261, '2020': 282, '2021': 286, '2022': 321}
    dfgab = pd.read_csv(str(ANO)+"dados/"+"gabarito.csv")
    dfgab['codigo'] = dfgab['codigo'].astype(int)
    gabarito = dfgab[dfgab['codigo'] == cod]['gabarito'].iloc[0]
    l = [row[df.columns[-1]] if pd.notnull(row[df.columns[-3]]) and row[df.columns[-3]] != '' else 0 for index, row in df.iterrows()]
    l.append(m[str(ANO)])
    l.append(gabarito)
    return l
def db2csv(nome):
    connection = sqlite3.connect('model_weights_biases'+str(nome)+'.db')
    cursor = connection.cursor()

    # Execute query
    cursor.execute('SELECT * FROM weights_biases')  # Replace 'your_table' with your actual table name

    # Fetch all rows and column headers
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]

    # Write to CSV, including column headers
    with open('weights_biases'+str(nome)+'.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(columns)  # Column headers
        csv_writer.writerows(rows)  # Data rows

    # Close connection
    connection.close()
def prever_nota_com_ia(acertos_decimal, ANO, cod):
    # Caminho para o diretório do modelo salvo
    model_directory = f"C:/Users/Marce/Codes/Frezza Fisica/models/{ANO}_{cod}"
    # Carregar o modelo
    modelo_carregado = load_model(model_directory)
    
    # Converter acertos decimais para binário e criar uma matriz para a previsão
    acertos_binario = format(int(acertos_decimal), '045b')  # Asegura-se de que o valor esteja em binário com 45 bits
    acertos_binario_array = np.array([[int(bit) for bit in acertos_binario]])  # Transforma em array do numpy

    # Fazer a previsão
    prediction = modelo_carregado.predict(acertos_binario_array)
    
    # Aqui, você pode ajustar a previsão conforme necessário, por exemplo, desnormalizando o valor se aplicável
    # Se você normalizou as notas durante o treinamento, faça a conversão inversa aqui
    nota_prevista = prediction.flatten()[0] * 1000  # Exemplo de desnormalização
    
    return nota_prevista
def prever_nota_com_ialote(lote, ANO, cod):
    # Caminho para o diretório do modelo salvo
    model_directory = f"C:/Users/Marce/Codes/Frezza Fisica/models/{ANO}_{cod}"
    # Carregar o modelo
    modelo_carregado = load_model(model_directory)
    
    # Converter acertos decimais para binário e criar uma matriz para a previsão
    # acertos_binario = format(int(acertos_decimal), '045b')  # Asegura-se de que o valor esteja em binário com 45 bits
    acertos_binario_array = np.array([lote])  # Transforma em array do numpy

    # Fazer a previsão
    prediction = modelo_carregado.predict(acertos_binario_array)
    
    # Aqui, você pode ajustar a previsão conforme necessário, por exemplo, desnormalizando o valor se aplicável
    # Se você normalizou as notas durante o treinamento, faça a conversão inversa aqui
    notas_previstas = [i * 1000 for i in prediction.flatten()] # Exemplo de desnormalização
    
    return notas_previstas
def calcular_pesos_ia(ANO, cod):
    s = 0
    q = []
    for i in range(45):
        m = achar_melhor_opcaoia(s, ANO, cod)
        q.append(m)
        s+= 2**(45-(m[0]))
    print(q)
    dados = q
    indices = list(range(len(dados)))
    primeiros = [i[0] for i in dados]
    segundos = [i[1] for i in dados]

    # Criando o gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(indices, primeiros, label='Primeiros Elementos', marker='o', linestyle='-', color='r')
    plt.plot(indices, segundos, label='Segundos Elementos', marker='x', linestyle='--', color='b')
    plt.title('Comparação dos Elementos das Tuplas')
    plt.xlabel('Índice da Tupla')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid(True)
    plt.show()
    return q
def analisarsigmoide(n, ANO, cod, p:bool=True):

    model_directory = f"C:/Users/Marce/Codes/Frezza Fisica/models/{str(ANO)}_{str(cod)}"
    # Carregar o modelo
    modelo_carregado = load_model(model_directory)
    # Número de variações para analisar
    n_variações = 10000

    # Inicializando um lote de entradas
    entradas = np.zeros((n_variações, 45))

    # Aqui, geramos variações para todas as sigmoides exceto a de interesse
    # Assumindo que a sigmóide de interesse é a de índice 10
    index_sigmóide_para_analisar = n

    # Exemplo: variando todas as outras sigmoides aleatoriamente para demonstração
    # Em uma aplicação real, essa parte pode ser mais específica ou sistemática
    np.random.seed(42)  # Para reprodutibilidade
    entradas[:, :index_sigmóide_para_analisar] = np.random.randint(0, 2, size=(n_variações, index_sigmóide_para_analisar))
    entradas[:, index_sigmóide_para_analisar+1:] = np.random.randint(0, 2, size=(n_variações, 44 - index_sigmóide_para_analisar))

    # Calculando as saídas com a sigmóide de interesse não invertida
    valores_saida_original = modelo_carregado.predict(entradas)

    # Invertendo a sigmóide de interesse e calculando as saídas novamente
    entradas[:, index_sigmóide_para_analisar] = 1  # Invertendo a sigmóide específica
    valores_saida_invertida = modelo_carregado.predict(entradas)

    # Calculando a diferença nas saídas devido à inversão da sigmóide de interesse
    diferenças = valores_saida_invertida - valores_saida_original

    # Plotando o gráfico
    if p:
        plt.figure(figsize=(10, 6))
        plt.scatter(valores_saida_original, diferenças, color='blue')
        plt.title('Impacto da Inversão da Sigmóide no Valor da Saída')
        plt.xlabel('Valor da Saída com Sigmóide Não Invertida')
        plt.ylabel('Diferença na Saída após Inversão')
        plt.grid(True)
        plt.show()
def criar_rarquivo(ANO, cod):
    df = pd.read_csv(f"{str(ANO)}dados/{str(cod)}.0.csv")
    df['Binario'] = df['Acertos_Decimal'].apply(dec2vet)
    bin_cols = df['Binario'].apply(lambda x: pd.Series(list(x))).astype(int)
    bin_cols.columns = [f'Resposta_{i+1}' for i in range(45)]
    new_df = pd.concat([df[df.columns[1]], bin_cols], axis=1)
    # new_df = df.sort_values(by="CO_POSICAO", ascending=True)[['NU_PARAM_A','NU_PARAM_B','NU_PARAM_C']]
    return new_df
def criar_cilindro(ANO, cod, n=30, p:bool=False):
    dados_totais = []
    model_directory = f"C:/Users/Marce/Codes/Frezza Fisica/models/{str(ANO)}_{str(cod)}"
    modelo_carregado = load_model(model_directory)

    # Carregar os dados
    # df = pd.read_csv(Rarquivo)
    df = criar_rarquivo(ANO, cod)
    df = df.drop(columns=df.columns[0])  # Remover a primeira coluna não booleana
    df = df.astype(bool)
    entradas = df.values  # Usar .values para obter um NumPy array

    # Função para calcular o valor da saída do modelo com e sem inversão da sigmóide
    def suavizar_com_media_movel(dados, tamanho_janela):
        if tamanho_janela < 2:
            return dados
        filtro = np.ones(tamanho_janela) / tamanho_janela
        dados_suavizados = np.convolve(dados, filtro, mode='same')
        return dados_suavizados

    # Função para calcular as diferenças de saída após inverter cada sigmóide
    def calcular_diferencas(entradas, modelo, index_sigmóide):
        entradas_modificadas = np.copy(entradas)
        entradas_modificadas[:, index_sigmóide] = 1 - entradas_modificadas[:, index_sigmóide]
        saida_original = modelo.predict(entradas).flatten()
        saida_modificada = modelo.predict(entradas_modificadas).flatten()
        return saida_original, saida_modificada - saida_original
    medias_moveis_salvas = []
    # Gerar gráficos para todas as sigmoides
    for index_sigmóide in range(entradas.shape[1]):
        saida_original, diferencas = calcular_diferencas(entradas, modelo_carregado, index_sigmóide)
        filtro_positivo = diferencas > 0
        saida_original = saida_original[filtro_positivo]
        diferencas = diferencas[filtro_positivo]
        # Ordenar os valores de saída e as diferenças
        indices_ordenados = np.argsort(saida_original)
        saida_original_ordenada = saida_original[indices_ordenados]
        diferencas_ordenadas = diferencas[indices_ordenados]
        
        # Aplicar suavização
        diferencas_suavizadas = suavizar_com_media_movel(diferencas_ordenadas, tamanho_janela=5000)
        medias_moveis_salvas.append(diferencas_suavizadas)
        limites = np.linspace(saida_original_ordenada.min(), saida_original_ordenada.max(), n + 1)

        # Inicializar listas para armazenar as médias e desvios padrões
        medias = []
        desvios = []

        # Calcular médias e desvios padrões para cada segmento
        for i in range(n):
            # Selecionar os dados dentro do segmento atual
            dentro_segmento = (saida_original_ordenada >= limites[i]) & (saida_original_ordenada < limites[i + 1])
            valores_segmento = diferencas_ordenadas[dentro_segmento]
            
            # Calcular média e desvio padrão
            media = valores_segmento.mean()
            desvio = valores_segmento.std()
            
            # Adicionar às listas
            medias.append(media)
            desvios.append(desvio)

        # Calcular os pontos médios dos limites para plotagem
        pontos_medios = (limites[:-1] + limites[1:]) / 2
        for ponto_medio, media, desvio in zip(pontos_medios, medias, desvios):
            dados_totais.append([index_sigmóide+1, ponto_medio, media, desvio])
        # Plotar as médias e os desvios padrões
        if p:
            plt.figure(figsize=(12, 7))
            plt.scatter(saida_original_ordenada, diferencas_ordenadas, alpha=0.3, label='Dados Originais', color='gray')
            plt.errorbar(pontos_medios, medias, yerr=desvios, fmt='o', color='blue', ecolor='lightgray', elinewidth=3, capsize=0)
            plt.title('Médias e Desvios Padrões para Segmentos do Espectro de Saída')
            plt.xlabel('Valor de Saída')
            plt.ylabel('Diferença')
            plt.grid(True)
            plt.show()
            # Plotar
            plt.figure(figsize=(12, 7))
            plt.scatter(saida_original_ordenada, diferencas_ordenadas, alpha=0.3, label='Diferenças Originais')
            plt.plot(saida_original_ordenada, diferencas_suavizadas, color='red', label='Diferenças Suavizadas', linewidth=2)
            plt.title(f'Impacto da Inversão da Sigmóide {index_sigmóide+1}')
            plt.xlabel('Valor de Saída Original')
            plt.ylabel('Diferença de Saída')
            plt.legend()
            plt.show()

    df_estatisticas_totais = pd.DataFrame(dados_totais, columns=['Sigmóide', 'Ponto_Medio', 'Media', 'Desvio_Padrao'])

    # Salvar o DataFrame em um arquivo CSV
    caminho_arquivo = f"estatisticas_medias_desvios_totais{str(ANO)}_{str(cod)}.csv"
    df_estatisticas_totais.to_csv(caminho_arquivo, index=False)

    print(f"Arquivo salvo em: {caminho_arquivo}")
def grafico_cilindro(ANO, cod, n=30):

    caminho_arquivo = f'estatisticas_medias_desvios_totais{str(ANO)}_{str(cod)}.csv'
    df = pd.read_csv(caminho_arquivo)

    print(f"Por favor, escolha um número de ponto entre 1 e {str(n)} (ou o máximo de pontos disponíveis):")
    n_esimo_ponto = int(input()) - 1  # Convertendo para base-0

    # Agrupar por 'Sigmóide' e pegar o n-ésimo registro de cada grupo, focando na coluna 'Media'
    medias_por_sigmóide_n_esimo_ponto = df.groupby('Sigmóide').nth(n_esimo_ponto)

    # Ordenar as médias por ordem decrescente para visualização
    medias_ordenadas = medias_por_sigmóide_n_esimo_ponto.sort_values(by='Media', ascending=False)
    print(medias_ordenadas)
    # Plotar o gráfico de barras
    plt.figure(figsize=(12, 8))
    plt.bar(medias_ordenadas['Sigmóide'].astype(str), medias_ordenadas['Media'], color='skyblue')
    plt.xlabel('Sigmóide')
    plt.ylabel('Média no ' + str(n_esimo_ponto + 1) + 'º Ponto')
    plt.title(f'Contribuição das Médias das Sigmoides no {n_esimo_ponto + 1}º Ponto (Ordenadas)')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Ajustar layout para evitar sobreposição de rótulos
    plt.show()
def determinar_nova_ordem_universal(df, p=11):
    # Filtrar o DataFrame para ignorar os primeiros 11 pontos
    df_filtrado = df[df['Posicao'] > p]
    
    # Para cada sigmóide, encontrar o índice máximo do ponto médio
    max_pos_por_sigmóide = df_filtrado.groupby('Sigmóide')['Ponto_Medio'].max()
    
    # Ordenar as sigmoides pelo índice do ponto médio máximo
    ordem_universal = max_pos_por_sigmóide.sort_values().index
    
    return ordem_universal
def grafico_cilindro_3d(ANO, cod, p=11):
    # caminho_arquivo = f'estatisticas_medias_desvios_totais{str(ANO)}_{str(cod)}.csv'
    # caminho_arquivo = f"resultado_final.csv"
    # df = pd.read_csv(caminho_arquivo)
    df = classificar(ANO, cod)
    # Adicionar coluna de Posição baseada no Ponto_Medio dentro de cada grupo de Sigmóide
    df['Posicao'] = df.groupby('Sigmóide').rank(method='first')['Ponto_Medio']

    # Ignorar os primeiros 11 pontos
    df = df[df['Posicao'] > p]

    # Determinar a ordem universal baseada no 15º ponto (agora efetivamente o 4º ponto no conjunto filtrado)
    ordem_universal = determinar_nova_ordem_universal(df, p)
    print(ordem_universal)
    
    fig = plt.figure(figsize=(120, 80))
    ax = fig.add_subplot(111, projection='3d')

    # Cores para as barras
    cores = plt.cm.jet(np.linspace(0, 1, len(ordem_universal)))

    # Número total de pontos disponíveis após ignorar os primeiros 11
    num_posicoes = int(df['Posicao'].max()) - p

    for i, sigmoide in enumerate(ordem_universal):
        # Filtrar os dados para a sigmóide atual e ordená-los pela posição
        df_sigmóide = df[df['Sigmóide'] == sigmoide].sort_values(by='Posicao')

        # Ajustar ypos para iniciar do 0 após ignorar os primeiros 11 pontos
        ypos = (df_sigmóide['Posicao'] - p-1)  # Ajustado para base-0 após ignorar os primeiros 11

        # Altura das barras
        dz = df_sigmóide['Media'].values

        # Posições das barras
        xpos = np.full(len(ypos), i)  # Fixo para cada sigmóide, varia com 'i'
        zpos = np.zeros(len(ypos))

        ax.bar3d(xpos, ypos, zpos, dx=0.8, dy=0.5, dz=dz, color=cores[i])

    ax.set_xlabel('Sigmóide')
    ax.set_ylabel('Posição')
    ax.set_zlabel('Média')
    ax.set_title('Contribuição das Médias das Sigmoides por Posição')

    # Ajustar os ticks e rótulos do eixo X para refletir a ordem universal
    ax.set_xticks(range(len(ordem_universal)))
    ax.set_xticklabels(ordem_universal)

    # Ajustar os ticks do eixo Y para mostrar as posições, começando do 12º ponto
    ax.set_yticks(range(num_posicoes))
    ax.set_yticklabels(range((p+1), (p+1) + num_posicoes))

    plt.tight_layout()
    plt.show()
def classificar(ANO, cod):
    # classificador_df = pd.read_csv("C:\\Users\Marce\\Downloads\\classificador_expanded.csv")
    estatisticas_df = pd.read_csv(f'estatisticas_medias_desvios_totais{str(ANO)}_{str(cod)}.csv')
    explodidor(ANO, cod)
    with open(f'nclassificador_{str(ANO)}_{str(cod)}.csv', 'r') as file:
        classificador_lines = file.readlines()

    # Processando o arquivo classificador.csv manualmente
    classificador_data = []
    for line in classificador_lines[1:]:  # Ignorando o cabeçalho
        parts = line.strip().split(',')
        area = parts[0]
        conteudo = parts[1]
        questoes = [int(q.strip()) - 90 for q in parts[2:] if q.strip().isdigit()]  # Ajustando os números das questões
        classificador_data.append((conteudo, questoes))

    # Convertendo para DataFrame e expandindo as questões para linhas separadas
    classificador_processed_df = pd.DataFrame(classificador_data, columns=['Conteudo', 'Questoes'])
    classificador_expanded = classificador_processed_df.explode('Questoes')

    # Criando um mapeamento de sigmoide para categoria
    sigmoide_para_categoria = pd.Series(classificador_expanded.Conteudo.values, index=classificador_expanded.Questoes).to_dict()

    # Lendo o arquivo de estatísticas
    # estatisticas_df = pd.read_csv('caminho_para_o_arquivo/estatisticas_medias_desvios_totais2021_910.csv')

    # Aplicando o mapeamento ao DataFrame de estatísticas
    estatisticas_df['Categoria'] = estatisticas_df['Sigmóide'].apply(lambda x: sigmoide_para_categoria.get(x))

    # Ajustando o Ponto_Medio para ser um inteiro entre 30 e 1, mantendo a ordem
    estatisticas_df['Ponto_Medio_Ajustado'] = estatisticas_df.groupby('Sigmóide').cumcount(ascending=True) + 1

    # Agrupar por Categoria e Ponto_Medio_Ajustado, somando as médias
    estatisticas_agrupadas_df = estatisticas_df.groupby(['Categoria', 'Ponto_Medio_Ajustado'])['Media'].sum().reset_index()

    # Renomear colunas para refletir a estrutura original
    estatisticas_agrupadas_df = estatisticas_agrupadas_df.rename(columns={'Ponto_Medio_Ajustado': 'Ponto_Medio'})
    estatisticas_agrupadas_df = estatisticas_agrupadas_df.rename(columns={'Categoria': 'Sigmóide'})

    # Salvar o resultado final, se necessário
    # estatisticas_agrupadas_df.to_csv('resultado_final.csv', index=False)
    return estatisticas_agrupadas_df
def explodidor(ANO, cod):
    caminho = f'classificador_{str(ANO)}_{str(cod)}.csv'
    classificador_df = pd.read_csv(caminho)

    expanded_rows_corrected = []

    # Iterar sobre cada linha do DataFrame classificador
    for _, row in classificador_df.iterrows():
        # Considerando a possibilidade de separação por ponto e vírgula e tratando "nan"
        questoes = str(row['questoes']).replace(';', ',').split(',')
        for questao in questoes:
            questao = questao.strip()  # Remover espaços em branco
            # Certificar-se de que a questão não está vazia e não é 'nan'
            if questao and questao.lower() != 'nan':  
                expanded_rows_corrected.append({
                    'area': row['area'],
                    'conteudo': row['conteudo'],
                    'questao': int(questao)
                })

    # Criar um novo DataFrame com as linhas corrigidas
    classificador_expanded_corrected_df = pd.DataFrame(expanded_rows_corrected)
    # return classificador_expanded_corrected_df
    # Salvar o DataFrame corrigido para um novo arquivo CSV
    expanded_corrected_csv_path = "n"+caminho
    classificador_expanded_corrected_df.to_csv(expanded_corrected_csv_path, index=False)

if __name__ == '__main__':
    t = time()
    # criar_cilindro(2018, 448)
    # criar_cilindro(2017, 392)
    # classificar(2021, 910)
    grafico_cilindro_3d(2021, 910)
    # explodidor()
    # classificar(2021, 910)
    print(time()-t)