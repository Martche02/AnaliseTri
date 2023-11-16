def criar(agrupamento, Csi, nquest=45, randomico=False):
    from funcoes_base import nota
    import random
    from more_itertools import distinct_permutations as idp
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize, to_rgb
    grupo = agrupamento
    Csi = [Csi,[[random.random(), random.random(), random.random()/4+0.1] for i in range(int(45/grupo))]][randomico]

    mapa = []
    quest = int(nquest/grupo)
    for i in range(quest):
        m = []
        for x in list(idp([1 if j<=i else 0 for j in range(quest)])):
            m.append([nota(x, [Csi[j] for j in range(len(Csi)) if j%grupo==int(grupo/2)], quest), str(x).rfind('1')+1-i])
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
    norm = Normalize(vmin=minimo, vmax=maximo)
    tamanho = -10
    # Para cada sub-lista em 'mapa'
    for i, sub_lista in enumerate(mapa):
        # Inicializa a base para o 'empilhamento' dos segmentos
        bottom = 0
        # Para cada valor na sub-lista
        for valor in sub_lista:
            # Escolhe a cor baseada na nota, passando o valor para o cmap
            cor = cmap(norm(valor[0]))
            # Ajusta a luminosidade baseada no potencial de ganho
            lum = pot_norm(valor[1])  # Normaliza o potencial de ganho
            cor = adjust_lightness(to_rgb(cor), lum)  # Ajusta a luminosidade
            # Plota uma 'barra' com a altura representando o valor, empilhada sobre a anterior
            ax.bar(i, tamanho/len(sub_lista), bottom=bottom, color=cor, width=1)
            # Atualiza a base para o próximo segmento
            bottom += tamanho/len(sub_lista)

    # Configura o eixo x para mostrar um label para cada sub-lista
    ax.set_xticks(range(len(mapa)))
    ax.set_xticklabels([f'{(i+1)*grupo}' for i in range(len(mapa))])
    ax.set_ylabel('Coerência')
    ax.set_xlabel('Acertos')

    # Cria o ScalarMappable e inicializa com o colormap e o objeto Normalize
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Você pode precisar definir o array se não estiver plotando qualquer mappable.

    # Adiciona a barra de cores ao gráfico, associada ao eixo ax
    cbar = plt.colorbar(sm, ax=ax)
    ticks = np.arange(0, 3000, 150)
    # tick_positions = norm(ticks)
    # cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(ticks)
    cbar.set_label('Nota *fora_de_escala', rotation=270, labelpad=15)
    plt.title('Nota por Acerto por Coerência Enem LC 2022')
    plt.gca().set_yticks([])
    normm = Normalize(vmin=0, vmax=1)  # Supondo que os valores de brilho variam de 0 a 1
    smm = plt.cm.ScalarMappable(cmap='gray', norm=normm)  # Usando a escala de cinza
    smm.set_array([])
    plt.colorbar(smm,ax=ax, orientation='vertical', label='Questões Fáceis')

    # Mostra o gráfico
    plt.savefig('meu_grafico.png')
    plt.show()