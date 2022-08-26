library(forecast)

df <- read.csv("./residuals/lm-final.csv")
df_jan22 <- read.csv("./residuals/jan-22.csv")

y <- df[,2]
y.ts <- as.ts(y)

y_jan <- df_jan22[,2]

# this is u on Laplace scale converted to ys scale
u <- 1.62
to_pred_next <- which(y_jan > u)
stopifnot(length(to_pred_next) == 45)

arima.forecasts <- list()
for (i in 1:(length(to_pred_next))) {
  print(paste0("i=", i, "/", length(to_pred_next)))
  ix <- to_pred_next[i]
  ys <- c(y, y_jan[1:ix])
  arima.forecasts[[i]] <- ys %>% auto.arima %>% forecast(h=1)
}

arima.point.forecasts <- rep(0, length(to_pred_next))
for (i in 1:(length(to_pred_next))) {
  arima.point.forecasts[i] <- arima.forecasts[[i]]$mean[1]
}

write.csv(arima.point.forecasts, "./data/arima-point-forecasts.csv")
