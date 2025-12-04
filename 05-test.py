import optuna
import numpy as np
import pandas as pd
import torch
from kan import KAN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from optuna.samplers import TPESampler
import warnings
from datetime import datetime
import requests
from io import BytesIO
import joblib

# Ignorar avisos
warnings.filterwarnings('ignore')

# Definir seed para reprodutibilidade
RANDOM_SEED = 27
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Verificar disponibilidade da GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Registrar ID da execução
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run ID: {run_id}")

# Carregar dados pré-processados
selected_features = ["S4_1", "S2_2", "S2_1", "S3_1"]
x_train = pd.read_csv("https://raw.githubusercontent.com/igorbzi/topicos_xl/main/projeto_IA_GEX1090/dados/processados/X_train_scaled.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/igorbzi/topicos_xl/main/projeto_IA_GEX1090/dados/processados/y_train_scaled.csv")

x_val = pd.read_csv("https://raw.githubusercontent.com/igorbzi/topicos_xl/main/projeto_IA_GEX1090/dados/processados/X_val_scaled.csv")
y_val = pd.read_csv("https://raw.githubusercontent.com/igorbzi/topicos_xl/main/projeto_IA_GEX1090/dados/processados/y_val_scaled.csv")

x_test = pd.read_csv("https://raw.githubusercontent.com/igorbzi/topicos_xl/main/projeto_IA_GEX1090/dados/processados/X_test_scaled.csv")
y_test = pd.read_csv("https://raw.githubusercontent.com/igorbzi/topicos_xl/main/projeto_IA_GEX1090/dados/processados/y_test_scaled.csv")

# Manter apenas features selecionadas
x_train = x_train[selected_features].copy()
x_val   = x_val[selected_features].copy()
x_test  = x_test[selected_features].copy()

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# Carregar scaler do y diretamente da URL
url_scaler = "https://raw.githubusercontent.com/igorbzi/topicos_xl/main/projeto_IA_GEX1090/dados/processados/scaler_y.pkl"
response = requests.get(url_scaler)
y_scaler = joblib.load(BytesIO(response.content))
print("✅ y_scaler carregado com sucesso")

# Converter para tensores PyTorch e mover para GPU
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)

x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)

x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

# Número de features
n_features = x_train.shape[1]
n_outputs = y_train.shape[1]

# Definir a função objetivo para o Optuna
def objective(trial):
    # Espaço de busca para hiperparâmetros
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 4)
    hidden_units = trial.suggest_int("hidden_units", 
                                    min(n_features, 32), 
                                    max(n_features, 32))
    
    grid_size = trial.suggest_int("grid_size", 5, 20)
    k = 3  # Fixando k=3 como sugerido
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "LBFGS"])
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True)
    
    # Construir a arquitetura da rede
    # [entrada, camadas_ocultas..., saída]
    width = [n_features]
    for _ in range(n_hidden_layers):
        width.append(hidden_units)
    width.append(n_outputs)
    
    # Criar e treinar o modelo KAN
    try:
        model = KAN(width=width, grid=grid_size, k=k).to(device)
        
        # Preparar o dataset no formato esperado pelo KAN
        dataset = {
            'train_input': x_train_tensor,
            'train_label': y_train_tensor,
            'test_input': x_val_tensor,
            'test_label': y_val_tensor
        }
        
        # Treinar o modelo
        result = model.fit(
            dataset=dataset,
            opt=optimizer_name,
            steps=30,  # Número reduzido para otimização mais rápida
            lamb=l2_reg,
            lr=learning_rate,
            batch=-1,  # Full batch
            log=1000
        )
        
        # Avaliar no conjunto de validação
        with torch.no_grad():
            val_preds = model(x_val_tensor).detach().cpu().numpy()
        
        # Calcular RMSE (métrica a ser minimizada)
        rmse = np.sqrt(mean_squared_error(y_val.values, val_preds))
        
        return rmse
    
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        # Retornar um valor alto para que o Optuna evite esta configuração
        return float('inf')

# Criar e executar o estudo Optuna
study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=30, show_progress_bar=True)

# Exibir os melhores hiperparâmetros encontrados
print("Melhor RMSE (val):", study.best_value)
print("Melhores parâmetros:")
for k, v in study.best_params.items():
    print(f"  - {k}: {v}")

# Salvar os resultados
trials_df = study.trials_dataframe()
trials_df.to_csv(f"kan_optuna_trials_{run_id}.csv", index=False)
pd.DataFrame([study.best_params]).to_csv(f"kan_best_params_{run_id}.csv", index=False)

# Treinar o modelo final com os melhores hiperparâmetros
best_params = study.best_params

# Construir a arquitetura da rede final
width = [n_features]
for _ in range(best_params["n_hidden_layers"]):
    width.append(best_params["hidden_units"])
width.append(n_outputs)

# Criar o modelo KAN final e mover para GPU
final_model = KAN(
    width=width, 
    grid=best_params["grid_size"], 
    k=3
).to(device)

# Preparar o dataset para o modelo final
dataset_final = {
    'train_input': x_train_tensor,
    'train_label': y_train_tensor,
    'test_input': x_val_tensor,
    'test_label': y_val_tensor
}

# Treinar o modelo final
result = final_model.fit(
    dataset=dataset_final,
    opt=best_params["optimizer"],
    steps=150,  # Mais épocas para o modelo final
    lamb=best_params["l2_reg"],
    lr=best_params["learning_rate"],
    batch=-1,  # Full batch
    log=500,    # Log a cada 500 passos
)

# Avaliar no conjunto de teste
with torch.no_grad():
    test_preds = final_model(x_test_tensor).detach().cpu().numpy()

# Calcular métricas
test_rmse = np.sqrt(mean_squared_error(y_test.values, test_preds))
test_mae = mean_absolute_error(y_test.values, test_preds)
test_r2 = r2_score(y_test.values, test_preds)

print(f"Teste -> RMSE: {test_rmse:.6f} | MAE: {test_mae:.6f} | R²: {test_r2:.6f}")

# Salvar o modelo final
torch.save(final_model.state_dict(), f"kan_final_model_{run_id}.pt")

# Converter previsões do KAN para a escala original usando o scaler carregado
y_test_mm = y_scaler.inverse_transform(y_test.values)
kan_pred_mm = y_scaler.inverse_transform(test_preds)

# Calcular métricas na escala original
kan_rmse_mm = np.sqrt(mean_squared_error(y_test_mm, kan_pred_mm))
kan_mae_mm = mean_absolute_error(y_test_mm, kan_pred_mm)
kan_r2_mm = r2_score(y_test_mm, kan_pred_mm)

print(f"Teste (mm) -> RMSE: {kan_rmse_mm:.3f} | MAE: {kan_mae_mm:.3f} | R²: {kan_r2_mm:.4f}")

# Mostrar primeiras 10 comparações
df_compare = pd.DataFrame({
    "Real_out1": y_test_mm[:,0],
    "Pred_out1": kan_pred_mm[:,0],
    "Real_out2": y_test_mm[:,1] if y_test_mm.shape[1] > 1 else None,
    "Pred_out2": kan_pred_mm[:,1] if kan_pred_mm.shape[1] > 1 else None,
})
print("\nPrimeiras 10 comparações (desnormalizadas):")
print(df_compare.head(10))

# Salvar métricas
metrics_mm_df = pd.DataFrame(
    [
        {"split": "test", "rmse": float(kan_rmse_mm), "mae": float(kan_mae_mm), "r2": float(kan_r2_mm)},
    ]
)
metrics_mm_df.to_csv(f"kan_final_metrics_mm_{run_id}.csv", index=False)

metrics_df = pd.DataFrame(
    [
        {"split": "test", "rmse": float(test_rmse), "mae": float(test_mae), "r2": float(test_r2)},
    ]
)
metrics_df.to_csv(f"kan_final_metrics_{run_id}.csv", index=False)

# Métricas por saída
print("\n===== MÉTRICAS POR SAÍDA (desnormalizadas) =====")

for i in range(n_outputs):
    mae = mean_absolute_error(y_test_mm[:, i], kan_pred_mm[:, i])
    mse = mean_squared_error(y_test_mm[:, i], kan_pred_mm[:, i])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_mm[:, i], kan_pred_mm[:, i])

    print(f"\nSaída {i+1}:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")