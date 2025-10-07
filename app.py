import streamlit as st
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1.5, 3.1, 3.6, 4.2, 4.8, 5.0, 5.6, 5.5, 6.3, 6.0]

def mean(values):
    return sum(values) / len(values)

def variance(values):
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)

def covariance(x, y):
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / len(x)

def predict(x):
    return b0 + b1 * x

b1 = covariance(x, y) / variance(x)

b0 = mean(y) - b1 * mean(x)

y_pred = [predict(xi) for xi in x]

mse = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred)) / len(y)
r2 = 1 - sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred)) / sum((yi - mean(y)) ** 2 for yi in y)


st.title("Regressão Linear Simples")
st.text("Regressão Linear Simples é uma técnica estatística usada para modelar a relação entre uma variável dependente e uma variável independente. O objetivo é encontrar a linha que melhor se ajusta aos dados.")
st.write("Demonstração utilizando uma base de dados fictícia de horas estudadas e notas obtidas.")
st.write("Os dados representam a relação entre o número de horas que um estudante estuda e a nota que ele obtém em um exame. (A nota predita é o resultado demonstrado abaixo)")

st.table({
    "Horas Estudadas": x,
    "Nota Obtida": y,
    "Nota Predita": y_pred
})

st.header("Cálculos Estatísticos")

st.subheader("Média")
st.text("A média representa o valor central dos dados. É usada como base para calcular a variância e a covariância.")
st.latex(r"\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}")
st.text(f"A média da base de dados do eixo X é {mean(x)} e do eixo Y é {mean(y):.2f}.")

st.subheader("Variância (Var(x))")
st.text("A variância mede a dispersão dos dados em relação à média. Uma variância maior indica que as horas estudadas variam bastante entre os indivíduos.")
st.text("Se a variância for pequena, os dados estão mais próximos da média. Se for grande, os dados estão mais espalhados.")
st.latex(r"Var(x) = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n}")
st.text(f"Obs: Como estamos utilizando todos os dados fictícios disponíveis, a variância é calculada com divisor n")
st.text(f"A variância da base de dados do eixo X é {variance(x)}.")

st.subheader("Covariância (Cov(x, y))")
st.text("A covariância indica a direção da relação linear entre duas variáveis. Se for positiva, ambas as variáveis tendem a aumentar juntas. Se for negativa, uma tende a aumentar enquanto a outra diminui.")
st.latex(r"Cov(x, y) = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{n}")
st.text(f"A covariância entre as variáveis X e Y é {covariance(x, y):.2f}.")
st.text(" Podemos observar que é um valor positivo, indicando que à medida que as horas estudadas aumentam, as notas obtidas também tendem a aumentar. Porém, o valor da covariância por si só não nos informa a força dessa relação, ela mostra apenas a direção, mas não a intesidade. Para entender a força da relação, precisamos olhar para o coeficiente de determinação (R²). que será abordado mais adiante.")

st.subheader("Coeficiente de Inclinação (b₁)")
st.text("Mostra a mudança esperada na variável dependente (nota obtida) para cada unidade de aumento na variável independente (horas estudadas).")
st.latex(r"b_1 = \frac{Cov(x, y)}{Var(x)}")
st.text(f"O coeficiente de inclinação (b1) é {b1:.2f}. Isso significa que para cada hora adicional estudada, a nota obtida aumenta em média {b1:.2f} pontos.")

st.subheader("Coeficiente Intercepto (b₀)")
st.text("Representa o valor esperado da variável dependente (nota obtida) quando a variável independente (horas estudadas) é 0.")
st.latex(r"b_0 = \bar{y} - b_1 \bar{x}")
st.text(f"O coeficiente intercepto (b0) é {b0:.2f}. Isso quer dizer que se um aluno não estudar nada, a nota esperada seria de {b0:.2f} pontos.")

st.subheader("Modelo ajustado")
st.latex(r"\hat{y} = b_0 + b_1 x => \hat{y} = " + f"{b0:.2f} + {b1:.2f}x")

st.header("Visualização Gráfica")
plt.scatter(x, y, color='blue', label='Dados Reais')
plt.plot(x, y_pred, color='red', label='Linha de Regressão')
plt.xlabel('Horas Estudadas')
plt.ylabel('Nota Obtida')
plt.title('Regressão Linear Simples')
plt.legend()

st.text("A linha vermelha representa a linha de regressão linear que melhor se ajusta aos dados reais. Ela é determinada pelos coeficientes b0 e b1 calculados anteriormente.")

info = f'β₀ = {b0:.2f}\nβ₁ = {b1:.2f}\nR² = {r2:.2f}\nMSE = {mse:.2f}'
plt.text(6.5, 2.0, info, fontsize=11, bbox=dict(facecolor='white', alpha=0.7))

st.pyplot(plt)

st.header("Métricas de Avaliação do Modelo")
st.subheader("Erro Quadrático Médio (MSE)")
st.text("O MSE mede a média dos quadrados dos erros, ou seja, a diferença média entre os valores reais e os valores preditos pelo modelo. Um MSE menor indica um modelo mais preciso.")
st.latex(r"MSE = \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}")
st.text(f"O Erro Quadrático Médio (MSE) do modelo é {mse:.2f}. Isso indica que em média, as previsões do modelo estão a aproximadamente {mse:.2f} pontos quadrados de distância dos valores reais.")

st.subheader("Coeficiente de Determinação (R²)")
st.text("Retomando ao R², por a Covariancia não ser padronizada e não dizer a força da relação, utilizamos o coeficiente de determinação R². Ele indica a proporção da variabilidade na variável dependente que pode ser explicada pela variável independente. Um R² próximo de 1 indica que o modelo explica bem os dados.")
st.latex(r"R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}")
st.text(f"O coeficiente de determinação (R²) do modelo é {r2:.2f}. Isso significa que aproximadamente {r2*100:.2f}% da variação nas notas obtidas pode ser explicada pelas horas estudadas.")

st.subheader("MSE (Mean Squared Error)")
st.text("O Erro Quadrático Médio é uma métrica que quantifica a diferença entre os valores reais e os valores preditos pelo modelo. Um MSE menor indica um modelo mais preciso.")
st.latex(r"MSE = \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}")
st.text(f"O Erro Quadrático Médio (MSE) do modelo é {mse:.2f}. Isso indica que em média, as previsões do modelo estão a aproximadamente {mse:.2f} pontos quadrados de distância dos valores reais.")

