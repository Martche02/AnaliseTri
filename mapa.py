from numpy import array, array_equal, arange
from numpy.polynomial.legendre import leggauss
from funcoes_base import nota
from random import random as random
from more_itertools import distinct_permutations as idp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_rgb
import time
import cProfile
import pstats
from multiprocessing import Pool
from multiprocessing import cpu_count

def analizar(fn):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = fn(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(15)
        return result
    return wrapper
def paralelizar(fn):
    def wrapper(lista_de_args):
        with Pool(processes=cpu_count()) as pool:
            result = pool.map(fn, lista_de_args)
        return result
    return wrapper
# def ma(bit_list):
#     output = 0
#     for bit in bit_list:
#         output = output * 2 + bit
#     return output
def find_last(lst, elm):
  gen = (len(lst) - 1 - i for i, v in enumerate(reversed(lst)) if v == elm)
  return next(gen, None)
def calcular_seta_m1(mapa, i, valor):
    resultados = []
    for n in range(len(valor[2])): # n para cada conteudo de valor2
        if valor[2][n] == 0: #se o item do indice n for 0, temos que obter a string adicionada
            ajuste = array([1 if i == n else 0 for i in range(len(valor[2]))])
            resultado = sorted(mapa[i + 1], key=lambda x: 1 if array_equal(x[2], valor[2] + ajuste) else 0)[0]
            resultados.append(resultado)
    return sorted(resultados, key=lambda x: x[0], reverse=True)[0] if 0 in valor[2] else []
def calcular_seta_m0(mapa, i, valor):
    resultados = []
    for n in range(len(valor[2])):
        if valor[2][n] == 1:
            ajuste = array([1 if i == n else 0 for i in range(len(valor[2]))])
            resultado = sorted(mapa[i - 1], key=lambda x: 1 if array_equal(x[2], valor[2] - ajuste) else 0)[0][3][0]
            resultados.append(resultado)
    return sorted(resultados, key=lambda x: x[0], reverse=True)[0]

# @analizar
def criar(agrupamento, Csi, nquest=45, randomico=False):
    Xq, A = leggauss(40)
    db = dict()
    grupo = agrupamento
    Csi = [Csi,[[random(), random(), random()/4+0.1] for i in range(int(45/grupo))]][randomico]
    mapa = []
    quest = int(nquest/grupo)
    for i in range(quest):
        m = []
        for x in list(idp([1 if j<=i else 0 for j in range(quest)])):
            t1 = time.time()
            m.append([nota(x, [Csi[j] for j in range(len(Csi)) if j%grupo==int(grupo/2)], Xq, A, quest, db), find_last(x,1)+1-i, array(x)])
            print(time.time()-t1)
        m = sorted(m, key=lambda x: x[0], reverse=True)
        # m.sort(reverse=True)
        mapa.append(m)

    # Suponha que 'mapa' seja uma lista de listas com seus dados
    # e que cada sub-lista de 'mapa' contenha valores entre 0 e 1

    # Cria a figura e o eixo do plot
    fig, ax = plt.subplots()

# Função para ajustar a luminosidade da cor
    def adjust_lightness(rgb, amount=0.5):
        import colorsys
        h, l, s = colorsys.rgb_to_hls(*rgb)
        l = max(0, min(1, amount * l))
        return colorsys.hls_to_rgb(h, 1-l, s)
    # Define o colormap
    cmap = plt.cm.viridis
    
    potenciais = [valor[1] for sublist in mapa for valor in sublist]
    pot_norm = Normalize(vmin=min(potenciais), vmax=max(potenciais))
    # Cria um objeto Normalize para mapear os valores de cor

    maximo = max(i[0] for i in mapa[0])
    minimo = min(i[0] for i in mapa[0])
    for j in mapa:
        maximo = max(max(i[0] for i in j),maximo)
        minimo = min(min(i[0] for i in j),minimo)
    print(maximo)
    print(minimo)
    norm = Normalize(vmin=minimo, vmax=maximo)
    tamanho = 10
    # Para cada sub-lista em 'mapa'
    for i, sub_lista in enumerate(mapa):
        # Inicializa a base para o 'empilhamento' dos segmentos
        bottom = 0
        # Para cada valor na sub-lista
        for j,valor in enumerate(sub_lista):
            # Escolhe a cor baseada na nota, passando o valor para o cmap
            cor = cmap(norm(valor[0]))
            # Ajusta a luminosidade baseada no potencial de ganho
            lum = pot_norm(valor[1])  # Normaliza o potencial de ganho
            cor = adjust_lightness(to_rgb(cor), lum)  # Ajusta a luminosidade
            # Plota uma 'barra' com a altura representando o valor, empilhada sobre a anterior
            ax.bar(i, tamanho/len(sub_lista), bottom=bottom, color=cor, width=1)
            # Atualiza a base para o próximo segmento
            bottom += tamanho/len(sub_lista)
            if i < quest:
                seta_m1 = calcular_seta_m1(mapa, i, valor)
                mapa[i][j].append([seta_m1])

            if i != 0 and i < quest: #arrumar isto para calcular para o ultimo
                seta_m0 = calcular_seta_m0(mapa, i, valor)
                mapa[i][j][3].append(seta_m0)
            # seta_m0 = max([ sorted(sub_lista, key=lambda x: x[2], reverse=True)[-1+2**len(valor[2])+ma( (valor[2]+np.array([1 if i==n else 0 for i in range(len(valor[2]))])))]    for n in range(len(valor[2])) if valor[2][n] == 0 and  ])
    # Configura o eixo x para mostrar um label para cada sub-lista
    ax.set_xticks(range(len(mapa)))
    ax.set_xticklabels([f'{(i+1)*grupo}' for i in range(len(mapa))])
    ax.set_ylabel('Coerência')
    ax.set_xlabel('Acertos')

    # Cria o ScalarMappable e inicializa com o colormap e o objeto Normalize
    sm = cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=norm)
    sm.set_array(arange(int(maximo), int(minimo), 150))  # Você pode precisar definir o array se não estiver plotando qualquer mappable.

    # Adiciona a barra de cores ao gráfico, associada ao eixo ax
    cbar = plt.colorbar(sm, ax=ax)
    # ticks = np.arange(int(maximo), int(minimo), 150)
    # tick_positions = norm(ticks)
    # cbar.set_ticks(tick_positions)
    # cbar.set_ticklabels(ticks)
    cbar.set_label('Nota', rotation=90, labelpad=15)
    plt.title('Nota por Acerto por Coerência Enem LC 2022')
    plt.gca().set_yticks([])
    normm = Normalize(vmin=0, vmax=1)  # Supondo que os valores de brilho variam de 0 a 1
    smm = plt.cm.ScalarMappable(cmap='gray', norm=normm)  # Usando a escala de cinza
    smm.set_array([])
    plt.colorbar(smm,ax=ax, orientation='vertical', label='Questões Fáceis')

    # Mostra o gráfico
    plt.savefig('meu_grafico.png')
    plt.show()