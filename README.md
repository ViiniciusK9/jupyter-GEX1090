
# Atividades da matéria de GEX1090 - TÓPICOS ESPECIAIS EM COMPUTAÇÃO XL (2025.2)

## Atividade 1: Análise Exploratoria de Dados (EDA)

Nesta primeira atividade da disciplina, vocês deverão realizar uma Análise Exploratória de Dados (EDA) sobre a base ‘df_full.csv’, que contém medições de sensores em um processo de separação trifásica de petróleo (óleo, água e gás). A tarefa tem como objetivo conhecer profundamente os dados, identificar possíveis problemas (valores ausentes, outliers, redundâncias) e levantar hipóteses que orientarão os próximos passos do projeto.

### Formato do Dataset

O arquivo fornecido possui a seguinte estrutura:

```csv

Timestamp,S1_1,S1_2,S1_3,S1_4,S1_5,S1_6,S2_1,S2_2,S3_1,S3_2,S4_1,S4_2,Nivel_agua,Nivel_total

28/03/2022 16:30:26.57745,1524.09200,1524.68570,1530.24903,1538.64225,1552.56175,1553.28008,1530.97697,1539.25600,1544.74917,1550.44190,1545.14272,1559.75003,257.0,257.0
```

Os dados são resultantes de um sistema de sensores com 4 canais S1-S4, sendo o canal 1 composto por 6 sensores S1_1, S1_2, ..., S1_6; e os canais 2, 3 e 4 compostos por 2 sensores cada. A numeração dos sensores inicia da extremidade inferior da fibra.

### Orientações para uso no Google Colab

1. Acesse o Google Colab em: [colab](https://colab.research.google.com)
2. Crie um novo notebook (Python 3).
3. Faça o upload do arquivo ‘df_full.csv’ para o ambiente do Colab (ou garanta acesso ao Drive).
4. Carregue os dados no Pandas.
5. Prossiga com as análises descritas a seguir.

### Etapas da Atividade

1. Carregamento e inspeção inicial
   - Verifique ‘df.shape’, ‘df.info()’ e visualize as primeiras linhas com ‘df.head()’.
   - Confirme se as colunas foram carregadas corretamente.
2. Resumo estatístico
   - Use ‘df.describe()’ para analisar média, mediana, desvio padrão, mínimo e máximo.
   - Avalie a variabilidade entre sensores.
3. Valores ausentes e duplicados
   - `df.isna().sum()` para identificar valores ausentes.
   - `df.duplicated().sum()` para verificar duplicatas.
4. Distribuição das variáveis
   - Crie histogramas para sensores (ex.: `df['S1_1'].hist(bins=30)`).
   - Utilize boxplots para detectar outliers (`df.boxplot(column=['S1_1'])`).
5. Correlação entre sensores
   - Calcule `df.corr()`. // selecione apenas as colunas dos sensores e das variáveis alvo.
   - Visualize com heatmap (ex.: Seaborn: `sns.heatmap(df.corr(), annot=True)`).
6. Análise temporal
   - Plote séries temporais de sensores.  Como cada sensor se comporta com o aumento do nível de água? E do nível total?
7. Relação com as variáveis-alvo
   - Análise `Nivel_agua` e `Nivel_total`.
   - Faça scatterplots entre sensores e níveis plt.scatter(df['Nivel_total'], df['S1_1'], alpha=0.5).

### Resultados Esperados

Ao final da atividade, o grupo deve entregar o notebook em formato PDF (`01_EDA_<grupo>.pdf`) contendo:

- Código estruturado e comentado.
- Gráficos (histogramas, boxplots, séries temporais, scatterplots, heatmaps).
- Análises textuais (pequenos parágrafos interpretando cada resultado).
- Um resumo final (5-10 linhas) destacando:
  - Principais padrões encontrados.
  - Problemas identificados (outliers, valores ausentes, redundância, desbalanceamento).
  - Hipóteses que podem orientar os próximos passos.

---

## Atividade 2: Pré-Processamento

Neste trabalho, você será responsável por implementar a etapa de pré
processamento de dados aplicada a um conjunto real de medições de sensores
ópticos. Esses sensores registram deslocamentos de comprimento de onda (em
nanômetros), relacionados aos níveis de água e de líquido total em um tanque
separador trifásico.

O objetivo principal é preparar a base de dados para que possa ser utilizada em
modelos de regressão que estimarão os níveis do tanque a partir das leituras dos
sensores. Para isso, você deverá seguir um roteiro de etapas fundamentais de pré
processamento, que incluem:

- Carregamento e checagem inicial dos dados
- Detecção e tratamento de outliers
- Divisão da base em treino, validação e teste
- Análise da distribuição das variáveis-alvo
- Normalização (scaling) das variáveis
- Seleção de features relevantes

Este processo é essencial para garantir que o modelo de aprendizado de máquina
trabalhe com dados consistentes, equilibrados e representativos do sistema físico
real.

Ao final, espera-se que você apresente um conjunto de dados limpo, normalizado
e reduzido às variáveis mais relevantes, pronto para ser utilizado na etapa de
modelagem.

### Passo 1: Carregamento e checagem inicial

A fim de garantir que os dados foram carregados corretamente e entender sua
estrutura.

1. Crie um novo notebook e nomeie-o como 02-PreProcessamento;
2. Carregue o arquivo df_full.csv;
3. Faça uma checagem básica: número de linhas e colunas, tipos de dados, presença de valores nulos, estatísticas descritivas iniciais;
4. Isso ajuda a identificar problemas logo no início e evitar erros.

### Passo 2: Tratamento de Outliers

Para identificar e lidar com pontos que estão muito fora do padrão normal dos
dados.

1. Detecte outliers com o algoritmo Local Outlier Factor (LOF);
   1. Configure 20 vizinhos.
   2. Defina a proporção de outliers esperada, como 5% ou 10%.
2. Crie uma cópia da base sem os outliers, para utilizar no restante do
processo.

### Passo 3: Dividir a Base de Dados em Treino/Validação/Teste

Com o objetivo de separar os dados em conjuntos independentes para treinar,
validar e testar o modelo.

1. Divida a base de dados. Divisão sugerida:
   1. Treino = 60%
   2. Validação = 20%
   3. Teste = 20%
2. Regras importantes:
   1. Sempre divida a base de dados antes de aplicar qualquer normalização.
   2. Para regressão, é importante que os conjuntos mantenham a mesma distribuição dos níveis (targets). Para isso, utilize estratificação por faixas (bins), por exemplo com pd.qcut.

### Passo 4: Verificar a Distribuição dos Níveis

Com o objetivo de garantir que os níveis de água e total (targets) estão bem
representados em diferentes faixas.

1. Faça gráficos ou tabelas para ver se os dados de treino têm valores em toda a faixa de interesse (baixo, médio e alto nível do tanque).
2. Se a distribuição estiver muito concentrada em apenas uma faixa, registre isso e discuta o impacto na previsão.

### Passo 5: Normalização / Scaling

Com o objetivo de colocar todas as variáveis (sensores e targets) na mesma escala para que o modelo aprenda de forma equilibrada.

1. Use o MinMaxScaler para transformar os valores para a faixa [0, 1].
2. Regras:
   1. Crie um scaler para X (sensores) e outro para y (targets).
   2. Faça o ajuste (fit) somente nos dados de treino (X_train, y_train).
   3. Aplique a transformação em X_val, X_test, y_val, y_test.

**Boas práticas:** Salve os scalers em arquivos (scaler_X.pkl e scaler_y.pkl). Isso garante que os mesmos parâmetros sejam usados futuramente, inclusive em produção.

### Passo 6: Seleção de Features

Com o objetivo de reduzir a dimensionalidade, melhorar desempenho e remover variáveis irrelevantes ou redundantes.

O RFECV (Recursive Feature Elimination with Cross-Validation) é uma técnica de seleção de variáveis.  Ele funciona removendo progressivamente as features menos importantes e avaliando o desempenho do modelo a cada etapa.

Atenção: o RFECV precisa de um modelo base para decidir quais variáveis são mais ou menos relevantes. Sugiro usar o modelo Random Forest como base porque:

1. Robustez: O Random Forest é menos sensível a ruídos e outliers.
2. Importância de variáveis: Ele calcula naturalmente o quanto cada feature contribui para a previsão.
3. Flexibilidade: Funciona bem em problemas de regressão e classificação.
4. Didática: É um modelo intuitivo para análise de importância de features.

Para implementação:

1. Defina um modelo de Random Forest Regressor.
2. Passe esse modelo como parâmetro para o RFECV.
3. Execute o RFECV com os dados de treino escalonados.

Observe:

1. Quais variáveis foram mantidas.
2. Qual foi o número ideal de features escolhido.
3. O desempenho do modelo durante a validação cruzada.

#### Boas práticas

- Não use os dados de teste nesta etapa. O teste deve ser reservado para a avaliação final.
- Ajuste o número de estimadores do Random Forest para ter resultados estáveis (por exemplo, 100 ou 200 árvores).
- Use cross-validation com k-folds (ex.: k=5) para reduzir a chance de resultados aleatórios.

---

## Atividade 3: Modelagem com Random Forest

Nesta etapa, você deverá aplicar o algoritmo Random Forest Regressor para
construir um modelo capaz de prever os níveis do separador trifásico a partir dos
dados tratados e preparados na fase de pré-processamento.

O objetivo é treinar o modelo, otimizar seus hiperparâmetros, avaliar seu
desempenho e interpretar os resultados obtidos.

### Passo 1: Configuração Inicial do Experimento

Antes de treinar o modelo:

1. Crie um novo notebook e nomeie-o como 03-RandomForest.
2. Carregue os dados já pré-processados (divididos em treino, validação e teste, com scaling aplicado).
3. Defina as métricas que serão usadas para avaliação. Para regressão, recomenda-se:
   1. RMSE (Root Mean Squared Error): erro médio quadrático em unidades do nível (cm).
   2. MAE (Mean Absolute Error): erro médio absoluto.
   3. R² (Coeficiente de Determinação): proporção da variância explicada.

### Passo 2: Treinamento do Modelo Baseline

1. Instancie um modelo de RandomForestRegressor com parâmetros padrão.
2. Treine-o apenas com os dados de treino.
3. Avalie seu desempenho em treino e validação.
4. Registre as métricas como referência inicial (baseline).

### Passo 3: Otimização de Hiperparâmetros

Para melhorar o desempenho do modelo, explore diferentes configurações. Os principais hiperparâmetros são:

- `n_estimators`: número de árvores (100, 200, 500).
- `max_depth`: profundidade máxima das árvores (ex.: None, 5, 10, 20).
- `min_samples_split`: número mínimo de amostras para dividir um nó.
- `min_samples_leaf`: número mínimo de amostras em cada folha.
- `max_features`: número de variáveis consideradas a cada divisão (ex.: auto, sqrt, log2).

1. Use técnicas de busca sistemática, como GridSearchCV ou RandomizedSearchCV, com validação cruzada (ex.: k=5), ou ainda o Optuna.
2. Registre as combinações testadas e as métricas de validação correspondentes.
3. Escolha o melhor conjunto de hiperparâmetros.

### Passo 4: Avaliação Final do Modelo

1. Treine novamente o Random Forest com os hiperparâmetros otimizados, usando treino + validação.
2. Avalie no conjunto de teste, que deve ser usado apenas agora.
3. Reporte os resultados finais (RMSE, MAE, R²) e compare com o baseline.

### Passo 5: Análise de Importância das Variáveis

O Random Forest permite extrair a importância relativa das features.

1. Gere o gráfico de importâncias das variáveis.
2. Interprete: quais sensores são mais relevantes para prever os níveis?
3. Discuta se o resultado faz sentido em relação ao conhecimento físico do sistema.

### Passo 6: Discussão dos Resultados

No relatório, discuta:

- O modelo otimizou significativamente em relação ao baseline?
- Houve sinais de overfitting (treino >> validação/teste)?
- As variáveis mais importantes coincidem com a intuição sobre o processo físico?
- Sugestões de melhorias: aumentar a base de dados, testar outros algoritmos (ex.: Gradient Boosting, XGBoost), aplicar técnicas de regularização.

#### Boas Práticas

- Fixe uma semente aleatória (random_state) para garantir reprodutibilidade.
- Documente as configurações testadas, mesmo as que não deram bons resultados.
- Evite usar o conjunto de teste antes da avaliação final.
- Salve o modelo final em arquivo (modelo_rf.pkl) para uso posterior.

---

## Atividade 4: Modelagem com Multilayer Perceptron (MLP)

### 1) Dados

Utilize o conjunto de dados normalizados (com MinMaxScaler), conforme utilizou na tarefa anterior.

### 2) Definição do Modelo MLP

O modelo MLPRegressor é uma rede neural feedforward de múltiplas camadas, com os seguintes parâmetros principais:

- n_hidden_layers: define o número de camadas ocultas.
- hidden_layers_size: define o número de neurônios em cada camada oculta.
- activation: função de ativação (ReLU, tanh, logistic, etc.).
- learning_rate_init: taxa de aprendizado inicial.
- solver: algoritmo de otimização (Podem testar com 'adam' e 'nadam')
- alpha: regularização L2.

Obs.: Regularização é uma técnica usada para evitar overfitting em modelos de aprendizado de máquina, adicionando uma penalização nos pesos na função de custo.

### 3) Otimização dos hiperparâmetros

Use o Optuna (ou Gridsearch) para otimizar os parâmetros do item 2. Use, por exemplo:

- `n_hidden_layers`: entre 1 e 4.
- `hidden_layers_size`: entre o número de neurônios de entrada (features) e 32.
- `activation`: ReLU e tanh.
- `learning_rate_init`: entre 1e-5 e 1e-2.
- `solver`:  adam e nadam
- `alpha`: regularização L2.

A regularização L2 é um hiperparâmetro que controla o “peso” da penalização. Se for muito pequeno, o modelo pode sobreajustar (overfit); se for muito grande, o modelo pode subajustar (underfit). Peça para o Optuna buscar o valor ideal dentro do intervalo 1e-6, 1e-2, por exemplo.

### 4) Treine o modelo final com os melhores hyperparâmetros e compare os resultados com o modelo Random Forest

---

## Atividade 5: Modelagem com Kolmogorov-Arnold-Networks (KAN)

Observações: Sigam o repositório [pykan](https://github.com/KindXiaoming/pykan) para instalar a biblioteca pykan.

### 1) Dados para o modelo

Utilize o mesmo conjunto de dados pré-processado e normalizado da atividade anterior.

### 2) Modelo KAN: Parâmetros importantes

- `grid_size`: o número de pontos da malha da spline (ex. 5-20).
- `k`: grau do B-spline (Ex. 2 ou 3).
- `hidden_layers`: número de camadas intermediárias: (Ex. 1-4).
- `hidden_units`: quantidade de neurônios por camada (Ex. 5).

--> Note que no pykan as camadas são definidas com "layers = [5, 11, 4, 2]". Neste exemplo, são 5 neurônios de entrada (5 sensores), duas camadas ocultas, uma com 11 e outra com 4 neurônios, e a camada de saída com 2 neurônios (ex. para nível de água e total).

- `otimizadores`: Adam, Nadam, LBGFS.
- `steps`: o número de épocas (ex. 50).
- `batch_size`: nem precisa configurar, em geral esse parâmetro é pouco relevante em KAN. É mais sensível à regularização, grid_size e k. O pykan atua por padrão com fullbatch.
- `learning_rate`: o número que controla o tamanho dos passos que o modelo dá ao ajustar seus parâmetros durante o treinamento.
- `regularização L2`: a técnica usada para evitar overfitting penalizando pesos muito grandes.

### 3) Otimização de hiperparâmetros

Utilize o Optuna (ou o otimizador que preferir) para encontrar o melhor conjunto de parâmetros.

Sugestão de busca:

- `hidden_layer`: 1 a 4
- `hidden_units`: entre o número e features e 32.
- `grid_size`: 5 a 20
- `k`: 3
- `learning_rate`: 1e-5 a 1e-2
- `otimizador`: Adam, Nadam, LBGFS
- `regularização L2`: 1e-6 a 1e-2

### 4) Treinamento e avaliação

Treine com "early stop" e usando o mesmo número de épocas que usou no MLP. Avalie o modelo usando **RMSE, MAE e r2** 

### 5) Análise

Compare o desempenho do KAN com Ranfom Forest e MLP.
Faça gráficos para apoiar a sua análise.

### 6) Entrega

Entregue o pdf do notebook no Sigaa, e se prepare para apresentar na aula do dia 09/12, em 10 minutos.
