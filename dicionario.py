import pandas as pd
from time import time
from multiprocessing import Pool
import os
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
def lerc(path:str, *args:str)->pd.DataFrame:
    return pd.read_csv(path, usecols=args, header=0, delimiter=';', engine="pyarrow")
def criarTabGab(ANO:int|str)->None:
    l = ["CO_POSICAO","SG_AREA", "CO_ITEM", "TX_GABARITO", "CO_PROVA"]
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
        gabarito = ''.join(pd.read_csv("2022dadositens/"+codigo+".csv").sort_values(by='CO_POSICAO')["TX_GABARITO"].tolist())
        return find_matching_responses_and_check_uniformity(respostas, gabarito, df)
def nota(ANO:int, prova:str, cod:str|int, respostas:str)->float|None:
    """ANO = 2022 | 2021 | 2020 etc.\n
    PROVA = LC | CH | CN | MT\n
    COD = 1999\n
    RESPOSTAS = 'abcdeabcdeabcdeabcde...abcde'"""
    original = pd.read_csv("2022dadositens/"+str(cod)+".csv").sort_values(by='CO_POSICAO')["CO_ITEM"].tolist()
    for i in os.listdir(str(ANO)+"dados/"):
        df = pd.read_csv(os.path.join(str(ANO)+"dados/", i))
        if df.columns[0][-2:] == prova:
            nova = pd.read_csv("2022dadositens/"+i[:4]+".csv").sort_values(by='CO_POSICAO')["CO_ITEM"].tolist()
            respostas_mapeadas = ''.join([dict(zip(original, respostas)).get(questao, "X") for questao in nova])
            n = nota_por_prova(2022,i[:4],respostas_mapeadas)
            if n is not None:
                return n
def process_file(args):
    ANO, prova, cod, respostas, filename = args
    original = pd.read_csv("2022dadositens/"+str(cod)+".csv").sort_values(by='CO_POSICAO')["CO_ITEM"].tolist()
    df = pd.read_csv(os.path.join(str(ANO)+"dados/", filename))
    if df.columns[0][-2:] == prova:
        nova = pd.read_csv("2022dadositens/"+filename[:4]+".csv").sort_values(by='CO_POSICAO')["CO_ITEM"].tolist()
        respostas_mapeadas = ''.join([dict(zip(original, respostas)).get(questao, "X") for questao in nova])
        n = nota_por_prova(2022, filename[:4], respostas_mapeadas)
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
if __name__ == '__main__':
    t = time()
    print(nota_parallel(2022, "CH", 1178, 'E'*44+'1'))
    print(f"time: {int(time()-t)} seg")