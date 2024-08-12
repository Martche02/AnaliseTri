# Carregar as bibliotecas necessárias
library(dplyr)
library(ggplot2)

# Carregar os parâmetros dos itens
parametros <- read.csv("C:\\Users\\Marce\\OneDrive\\Documentos\\Rarquivop.csv")

# Organizar os parâmetros dos itens
parametros <- parametros %>%
  arrange(CO_POSICAO) %>%
  select(NU_PARAM_A, NU_PARAM_B, NU_PARAM_C)

# Carregar as respostas dos alunos
respostas_alunos <- read.csv("C:\\Users\\Marce\\OneDrive\\Documentos\\Rarquivo.csv")

# Selecionar as primeiras 20 entradas
respostas_selecionadas <- respostas_alunos[1:2000,]

# Remover a coluna de notas reais
notas_reais <- respostas_selecionadas$NU_NOTA_CN
respostas_selecionadas <- respostas_selecionadas %>% select(-NU_NOTA_CN)

# Função para calcular EAP (Expected a Posteriori) a partir de parâmetros arbitrários
calcular_eap <- function(respostas, parametros, mean_theta = 501.15, sd_theta = 113.1) {
# calcular_eap <- function(respostas, parametros, mean_theta = 500, sd_theta = 100) {
  # Definir uma grade de pontos de theta
  grid_points <- seq(-4, 4, length.out = 100)
  
  # Função de verossimilhança para a estimação de theta
  calculate_likelihood <- function(theta, respostas, parametros) {
    a <- parametros$NU_PARAM_A
    b <- parametros$NU_PARAM_B
    g <- parametros$NU_PARAM_C
    p <- g + (1 - g) / (1 + exp(-a * (theta - b)))
    likelihood <- respostas * log(p) + (1 - respostas) * log(1 - p)
    return(exp(sum(likelihood, na.rm = TRUE)))
  }
  
  # Calcular a verossimilhança para cada ponto na grade
  likelihoods <- sapply(grid_points, calculate_likelihood, respostas = respostas, parametros = parametros)
  
  # Distribuição a priori (assumindo normal padrão)
  prior_distribution <- dnorm(grid_points)
  
  # Calcular a distribuição a posteriori
  posterior <- likelihoods * prior_distribution
  posterior <- posterior / sum(posterior, na.rm = TRUE)
  
  # Calcular EAP (Expected a Posteriori)
  eap <- sum(grid_points * posterior, na.rm = TRUE)
  
  # Padronizar a estimativa de EAP
  eap_standardized <- mean_theta + sd_theta * eap
  
  return(eap_standardized)
}

# Transformar as respostas selecionadas em uma matriz
respostas_matriz <- as.matrix(respostas_selecionadas)

# Calcular as habilidades estimadas (notas) para as respostas selecionadas
notas_estimadas <- apply(respostas_matriz, 1, calcular_eap, parametros = parametros)

# Comparar as habilidades estimadas com as notas reais
comparacao <- data.frame(
  Nota_Real = notas_reais,
  Nota_Estimada = notas_estimadas
)

# Calcular o erro de aproximação
comparacao$Erro <- comparacao$Nota_Real - comparacao$Nota_Estimada

# Exibir a comparação
print(comparacao)

# Calcular o erro médio absoluto (MAE)
mae <- mean(abs(comparacao$Erro), na.rm = TRUE)
print(paste("Erro Médio Absoluto (MAE):", mae))

# Criar o gráfico
ggplot(comparacao, aes(x = Nota_Real, y = Erro)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Erro de Aproximação vs Nota Real",
       x = "Nota Real",
       y = "Erro de Aproximação (Nota Real - Nota Estimada)") +
  theme_minimal()
