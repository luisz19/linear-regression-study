import streamlit as st
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1.5, 3.1, 3.6, 4.2, 4.8, 5.0, 5.6, 5.5, 6.3, 6.0]

def mean(values):
    return sum(values) / len(values)

def variance(values):
    m = mean(values)
    return sum((x - m) ** 2 for x in values)

def covariance(x, y):
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))

def predict(x):
    return b0 + b1 * x

b1 = covariance(x, y) / variance(x)

b0 = mean(y) - b1 * mean(x)

y_pred = [predict(xi) for xi in x]

mse = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred)) / len(y)
r2 = 1 - sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred)) / sum((yi - mean(y)) ** 2 for yi in y)

for xi, yi in zip(x, y):
   print(f"horas estudadas: {xi}, nota obtida: {yi}")

st.table({
    "Horas Estudadas": x,
    "Nota Obtida": y,
    "Nota Predita": y_pred
})

st.write(f"Média: xm = {mean(x)}, ym = {mean(y)}")
st.write(f"Variância: Var(x) = {variance(x)}")
st.write(f"Covariância: Cov(x, y) = {covariance(x, y)}")
st.write(f"Coeficiente de inclinação (b1): {b1}")
st.write(f"Coeficiente intercepto (b0): {b0}")
st.write(f"Erro Quadrático Médio (MSE): {mse}")
st.write(f"Coeficiente de Determinação (R²): {r2}")


plt.scatter(x, y, color='blue', label='Dados Reais')
plt.plot(x, y_pred, color='red', label='Linha de Regressão')
plt.xlabel('Horas Estudadas')
plt.ylabel('Nota Obtida')
plt.title('Regressão Linear Simples')
plt.legend()

info = f'β₀ = {b0:.2f}\nβ₁ = {b1:.2f}\nR² = {r2:.2f}\nMSE = {mse:.2f}'
plt.text(6.5, 2.0, info, fontsize=11, bbox=dict(facecolor='white', alpha=0.7))

st.pyplot(plt)
