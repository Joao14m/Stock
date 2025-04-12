import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Baixar os dados da AAPL com dividendos
df = yf.download("AAPL", start="2020-01-01", end="2025-01-01", actions=True)

# Criar coluna de alvo: 1 se o preço subir no dia seguinte
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Calcular média móvel de 5 dias e retornos diários
df["MA5"] = df["Close"].rolling(window=5).mean()
df["Returns"] = df["Close"].pct_change()

# Remover linhas com valores ausentes
df = df.dropna()

# Definir variáveis
X = df[["MA5", "Returns"]]
y = df["Target"]

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Treinar modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Previsão
df["Predicted"] = model.predict(X)
df["Strategy"] = df["Predicted"].shift(1) * df["Returns"]

# Inicializando o número de ações (supondo que você começou com 1 ação, por exemplo)
df["Shares"] = 1

# Inicializando o saldo de dividendos acumulados
df["Accumulated_Dividends"] = 0

# Calculando os dividendos e acumulando
for i in range(1, len(df)):
    # Dividendos recebidos no dia
    dividend_received = df["Dividends"].iloc[i-1]
    
    # Atualizando o saldo de dividendos acumulados
    df["Accumulated_Dividends"].iloc[i] = df["Accumulated_Dividends"].iloc[i-1] + dividend_received
    
    # Verificando se os dividendos acumulados são suficientes para comprar mais ações
    # Agora com a comparação correta usando .iloc[i] para acessar o valor escalar
    if df["Accumulated_Dividends"].iloc[i] >= df["Close"].iloc[i]: 
        # Calculando quantas ações podemos comprar com os dividendos acumulados
        new_shares = df["Accumulated_Dividends"].iloc[i] // df["Close"].iloc[i]
        
        # Atualizando o número de ações
        df["Shares"].iloc[i] = df["Shares"].iloc[i-1] + new_shares
        
        # Atualizando os dividendos acumulados após a compra das ações
        df["Accumulated_Dividends"].iloc[i] -= new_shares * df["Close"].iloc[i]
    else:
        # Se não tiver o suficiente para comprar mais ações, mantemos o número de ações
        df["Shares"].iloc[i] = df["Shares"].iloc[i-1]

# Calculando o retorno da estratégia considerando o reinvestimento de dividendos
df["Strategy"] = (df["Predicted"].shift(1) * df["Returns"] * df["Shares"])

# Calculando o retorno cumulativo com a estratégia de reinvestimento
df["Cumulative"] = (1 + df["Strategy"]).cumprod()

# Reinvestimento de dividendos no retorno do mercado (sem mudanças)
df["Dividend_Yield"] = df["Dividends"] / df["Close"].shift(1)
df["Market_Returns"] = df["Returns"] + df["Dividend_Yield"]
df["Market"] = (1 + df["Market_Returns"]).cumprod()

# Plotar gráfico
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Cumulative"], label="Model Strategy", color="blue")
plt.plot(df.index, df["Market"], label="Market w/ Reinvested Dividends", color="orange")
plt.title("Model Strategy vs Market with Dividend Reinvestment")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Mostrar acurácia
print("Model Accuracy:", model.score(X_test, y_test))
