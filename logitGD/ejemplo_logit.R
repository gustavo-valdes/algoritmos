## Librerias
library(Rcpp) # Para cargar código en C++. sourceCpp

# Cargamos función para estimar el modelo con descenso del gradiente
sourceCpp("logitGD.cpp")
# El objetivo de este código es la estimación y predicción para modelos con una gran cantidad
# de variables explicativas, no calcula las varianzas de los coeficientes.

## Datos ficticios
N <- 2000 # Num. de observaciones
K <- 1000 # Num. de variables explicativas (no incluye constante)

set.seed(1)
X    <- rnorm(n = N*K, mean = 0, sd = .5)
X    <- cbind(1, matrix(X, nrow = N))

set.seed(2)
beta <- rnorm(n = K+1, mean = 0, sd = .5) # K + 1 ya que incluye una constante
beta <- matrix(beta, nrow = K+1)

prob <- 1 / (1 + exp(- X %*% beta)) # P(y = 1 | x)

Y    <- unlist(lapply(prob, function(x) rbinom(1, 1, x)))

## Logit
set.seed(3)
beta_0 <- rnorm(n = K+1, mean = 0, sd = .5) # Valores iniciales

fit <- logitGD(Y = Y, X = X, init_val = beta, Ytest = Y, Xtest = X)
summary(fit$beta - beta)
