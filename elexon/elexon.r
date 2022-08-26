library(rje)
library(ggplot2)
library(readr)
library(magrittr)
library(tibble)
library(tidyverse)
library(timeDate)
library(mgcv)
library(mgcViz)
library(latex2exp)
library(grid)
library(tseries)
library(ExtDist)

a_coef <- 91

rel_amnt <- 1.5

weekdays_lvls <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                   "Saturday", "Sunday")

#### viz
example_plot <- function() {
  df <- read_csv("data/elexon-data-2021.csv")
  df %<>% select(settlementPeriod, settlementDate, imbalancePriceAmountGBP)
  christmas_2021 <- c("2021-12-24", "2021-12-25", "2021-12-26", "2021-12-27")
  christmas_2021 <- as.Date(christmas_2021)
  christmas_df <- df[df$settlementDate %in% christmas_2021,]
  christmas_df$settlementDate <- as.factor(christmas_df$settlementDate)

  names(christmas_df) <- c("Period","Date","Imbalance")

  plt <- ggplot(christmas_df, aes(x=Period, y=Imbalance)) +
          geom_line(aes(col=Date)) +
          geom_point(aes(col=Date)) +
          scale_x_continuous(breaks=seq(4, 48, by=4)) +
          theme(legend.position=c(.8,.85),axis.text=rel(2.0),axis.title=rel(2.0))
  #print(plt)
  ggsave("figures/christmas_2021_imbalance.svg", plot=plt, height=1500, width=1800, units="px")
}

get_nice_df_ <- function(df) {
  is_holiday <- function(date_str) {
    td <- timeDate(date_str)
    timeDate::isHoliday(td, holidays=GBBankHoliday())
  }
  ## prices are in MWh
  df$is_holiday = df$settlementDate %>% is_holiday
  df$weekday = df$settlementDate %>% weekdays
  df$weekday = factor(df$weekday, levels = weekdays_lvls)

  df$period = factor(df$settlementPeriod, levels=1:50)

  df$week_number <- df$settlementDate %>%
                    as.character %>%
                    strftime(., format="%V") %>%
                    as.integer
  df$year <- df$settlementDate %>% format("%Y") %>% as.integer %>% as.factor
  df$month <- month.abb[df$settlementDate %>% format("%m") %>% as.integer] %>%
              factor(., levels=month.abb)
  df$day <- df$settlementDate %>% format("%d") %>% as.integer

  df <- arrange(df, settlementDate, settlementPeriod)
  df

}

get_nice_df <- function() {
  df <- read_csv("data/elexon-clean-2021.csv")
  df <- get_nice_df_(df)
  df
}

plot_whole_year <- function(df) {
  first_of_month_period_1 <- (df$day == 1) & (df$period == 1)
  xticks <- ifelse(first_of_month_period_1,
                   df$settlementDate %>% months,
                   "")
  df$Time <- 1:nrow(df)
  plt1 <- ggplot(df) +
          geom_line(aes(y=imbalancePriceAmountGBP, x=Time)) +
          scale_x_continuous(breaks=which(xticks!=""), labels=xticks[xticks!=""]) +
          theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
                axis.text.y=element_text(size=rel(rel_amnt)),
                axis.title.y=element_text(size=rel(rel_amnt)),
                ) +
          ylab("Imbalance (£/MWh)")
  plt2 <- ggplot(df) +
          geom_line(aes(y=log(imbalancePriceAmountGBP + a_coef), x=Time)) +
          scale_x_continuous(breaks=which(xticks!=""), labels=xticks[xticks!=""]) +
          theme(axis.title.x=element_blank(),
                axis.text.x=element_text(angle=30,size=rel(1.2)),
                axis.title.y=element_text(size=rel(rel_amnt)),
                ) +
          ylab(paste0("log(Imbalance + ", as.character(a_coef),")"))
  #plt2 <- ggplot(df) +
          #geom_line(aes(y=log(1+imbalancePriceAmountGBP - min(imbalancePriceAmountGBP, na.rm=T)), x=Time)) +
          #scale_x_continuous(breaks=which(xticks!=""), labels=xticks[xticks!=""]) +
          #theme(axis.title.x=element_blank(),axis.text.x=element_text(angle=30),
                #) +
          #ylab("log(1 + Imbalance - min(Imbalance))")


  #pdf("figures/imbalance_whole_year.pdf", width=14, height=10)
  pdf("figures/imbalance_whole_year.pdf")
  grid.newpage()
  plt <- grid.draw(rbind(ggplotGrob(plt1),ggplotGrob(plt2), size="first"))
  ## ggsave doesn't work with grid...
  #ggsave(file="figures/imbalance_whole_year.pdf", plot=plt, height=1500, width=1800, units="px")
  print(plt)
  dev.off()
}

#### model fitting (we start modifying the df quite heavily now, e.g. normalizing etc)
get_features_df <- function(df) {
  df <- df %>% select(period, imbalancePriceAmountGBP,
                      month, weekday, is_holiday, day,
                      imbalanceQuantityMAW,
                      settlementDate) %>%
          rename(price=imbalancePriceAmountGBP, date=settlementDate,
                 quantity=imbalanceQuantityMAW)

  ## impute values for June 23rd, not sure why these are missing
  df[(df$date == "2021-06-23") & (df$period==37),]$price <-
    mean(df[(df$month == factor("Jun", levels=month.abb)) & (df$period!=37),]$price)
  df[(df$date == "2021-06-23") & (df$period==34),]$quantity <-
    mean(df[(df$month == factor("Jun", levels=month.abb)) & (df$period!=34),]$quantity)
  stopifnot(df$price %>% (function(x){!is.na(x)}) %>% all)

  # remove period 49,50
  stopifnot(nrow(df$date == "2021-10-31") == 50)
  df <- df[!((df$period == 50) | (df$period == 49)),]
  df$period <- factor(df$period, levels=1:48)
  stopifnot(nrow(df$date == "2021-10-31") == 48)


  ## impute values for period 47, 48
  stopifnot(nrow(df$date == "2021-03-28") == 46)
  new_47_df <- df[(df$date == "2021-03-28") & (df$period==46),]
  new_48_df <- df[(df$date == "2021-03-28") & (df$period==46),]
  stopifnot(nrow(df$date == "2021-03-28") == 48)


  new_47_df$period[1] <- factor(47, levels=1:48)
  new_48_df$period[1] <- factor(48, levels=1:48)
  new_47_df$price[1] <- mean(df[(df$period == 47) & (as.character(df$month) == "Mar"),]$price)
  new_48_df$price[1] <-mean(df[(df$period == 48) & (as.character(df$month) == "Mar"),]$price)
  new_47_df$quantity[1] <- mean(df[(df$period == 47) & (as.character(df$month) == "Mar"),]$quantity)
  new_48_df$quantity[1] <- mean(df[(df$period == 48) & (as.character(df$month) == "Mar"),]$quantity)

  df <- rbind(df, new_47_df, new_48_df)
  df <- df[order(df$date, df$period),]
  stopifnot(nrow(df$date == "2021-03-28") == 48)

  ## check we've removed all NAs. unfortunately there's nothing we can do about Jan 1st,
  ## since we don't have data for Dec 31st (and we are calculating prev day values)
  stopifnot(!(df[df$date != "2021-01-01",] %>% is.na %>% any))

  ## now that we've imputed values, create log cols
  #df$log_price <- log(df$price -(min(df$price) - 1))
  df$log_price <- log(df$price + a_coef)

  n <- nrow(df)
  df$last_price1 <- NA
  df$last_price1[2:n] <- head(df$price, -1)
  df$last_price2 <- NA
  df$last_price2[3:n] <- head(df$price, -2)
  df$last_price3 <- NA
  df$last_price3[4:n] <- head(df$price, -3)
  df$last_log_price1 <- NA
  df$last_log_price1[2:n] <- head(df$log_price, -1)
  df$last_log_price2 <- NA
  df$last_log_price2[3:n] <- head(df$log_price, -2)
  df$last_log_price3 <- NA
  df$last_log_price3[4:n] <- head(df$log_price, -3)


  df$last_quantity1 <- NA
  df$last_quantity1[2:n] <- head(df$quantity, -1)
  df$last_quantity2 <- NA
  df$last_quantity2[3:n] <- head(df$quantity, -2)
  df$last_quantity3 <- NA
  df$last_quantity3[4:n] <- head(df$quantity, -3)

  df$yday_price <- NA
  df$yday_log_price <- NA
  df$yday_quantity <- NA
  ## this is bad, but it works
  for (i in 49:n) {
    d <- df$date[i]
    prev_d <- d - 1
    p <- df$period[i]
    yday_ix <- which((df$date == prev_d) & (df$period == p))
    if (length(yday_ix) > 0) {
      df$yday_price[i] <- df$price[yday_ix]
      df$yday_quantity[i] <- df$quantity[yday_ix]
      df$yday_log_price[i] <- df$log_price[yday_ix]
    } else if (length(yday_ix) == 0) {
      print(paste0("can't find yday ix for day = ", d, " period = ", p))
    } else {
      print(paste0("ERROR, day=", d, ", period= ", p))
    }
  }

  df
}

normalize_features <- function(df) {

  price_mean <- mean(df$price, na.rm=T)
  quantity_mean <- mean(df$quantity, na.rm=T)
  log_price_mean <- mean(df$log_price, na.rm=T)
  price_sd <- sd(df$price, na.rm=T)
  quantity_sd <- sd(df$quantity, na.rm=T)
  log_price_sd <- sd(df$log_price, na.rm=T)

  df$price <- (df$price - price_mean) / price_sd
  df$last_price1 <- (df$last_price1 - price_mean) / price_sd
  df$last_price2 <- (df$last_price2 - price_mean) / price_sd
  df$last_price3 <- (df$last_price3 - price_mean) / price_sd
  df$yday_price <- (df$yday_price - price_mean) / price_sd

  df$log_price <- (df$log_price - log_price_mean) / log_price_sd
  df$last_log_price1 <- (df$last_log_price1 - log_price_mean) / log_price_sd
  df$last_log_price2 <- (df$last_log_price2 - log_price_mean) / log_price_sd
  df$last_log_price3 <- (df$last_log_price3 - log_price_mean) / log_price_sd
  df$yday_log_price <- (df$yday_log_price - log_price_mean) / log_price_sd

  df$quantity <- (df$quantity - quantity_mean) / quantity_sd
  df$last_quantity1 <- (df$last_quantity1 - quantity_mean) / quantity_sd
  df$last_quantity2 <- (df$last_quantity2 - quantity_mean) / quantity_sd
  df$last_quantity3 <- (df$last_quantity3 - quantity_mean) / quantity_sd
  df$yday_quantity <- (df$yday_quantity - quantity_mean) / quantity_sd

  df
}

get_model_fitting_df <- function() {
  df <- get_nice_df()
  df <- get_features_df(df)
  df <- normalize_features(df)
  df
}

## here df is a new training df
residualize_df <- function(df, train_df) {

  df <- df %>% select(period, imbalancePriceAmountGBP,
                      month, weekday, is_holiday, day,
                      imbalanceQuantityMAW,
                      settlementDate) %>%
          rename(price=imbalancePriceAmountGBP, date=settlementDate,
                 quantity=imbalanceQuantityMAW)

  df$log_price <- log(df$price + a_coef)

  n <- nrow(df)
  df$last_price1 <- NA
  df$last_price1[2:n] <- head(df$price, -1)
  df$last_price2 <- NA
  df$last_price2[3:n] <- head(df$price, -2)
  df$last_price3 <- NA
  df$last_price3[4:n] <- head(df$price, -3)
  df$last_log_price1 <- NA
  df$last_log_price1[2:n] <- head(df$log_price, -1)
  df$last_log_price2 <- NA
  df$last_log_price2[3:n] <- head(df$log_price, -2)
  df$last_log_price3 <- NA
  df$last_log_price3[4:n] <- head(df$log_price, -3)


  df$last_quantity1 <- NA
  df$last_quantity1[2:n] <- head(df$quantity, -1)
  df$last_quantity2 <- NA
  df$last_quantity2[3:n] <- head(df$quantity, -2)
  df$last_quantity3 <- NA
  df$last_quantity3[4:n] <- head(df$quantity, -3)

  df$yday_price <- NA
  df$yday_log_price <- NA
  df$yday_quantity <- NA
  ## this is bad, but it works
  for (i in 49:n) {
    d <- df$date[i]
    prev_d <- d - 1
    p <- df$period[i]
    yday_ix <- which((df$date == prev_d) & (df$period == p))
    if (length(yday_ix) > 0) {
      df$yday_price[i] <- df$price[yday_ix]
      df$yday_quantity[i] <- df$quantity[yday_ix]
      df$yday_log_price[i] <- df$log_price[yday_ix]
    } else if (length(yday_ix) == 0) {
      print(paste0("can't find yday ix for day = ", d, " period = ", p))
    } else {
      print(paste0("ERROR, day=", d, ", period= ", p))
    }
  }

  price_mean <- mean(train_df$price, na.rm=T)
  quantity_mean <- mean(train_df$quantity, na.rm=T)
  log_price_mean <- mean(train_df$log_price, na.rm=T)
  price_sd <- sd(train_df$price, na.rm=T)
  quantity_sd <- sd(train_df$quantity, na.rm=T)
  log_price_sd <- sd(train_df$log_price, na.rm=T)
  print(paste0("log_price_mean: ", log_price_mean))
  print(paste0("log_price_sd: ", log_price_sd))

  df$price <- (df$price - price_mean) / price_sd
  df$last_price1 <- (df$last_price1 - price_mean) / price_sd
  df$last_price2 <- (df$last_price2 - price_mean) / price_sd
  df$last_price3 <- (df$last_price3 - price_mean) / price_sd
  df$yday_price <- (df$yday_price - price_mean) / price_sd

  df$log_price <- (df$log_price - log_price_mean) / log_price_sd
  df$last_log_price1 <- (df$last_log_price1 - log_price_mean) / log_price_sd
  df$last_log_price2 <- (df$last_log_price2 - log_price_mean) / log_price_sd
  df$last_log_price3 <- (df$last_log_price3 - log_price_mean) / log_price_sd
  df$yday_log_price <- (df$yday_log_price - log_price_mean) / log_price_sd

  df$quantity <- (df$quantity - quantity_mean) / quantity_sd
  df$last_quantity1 <- (df$last_quantity1 - quantity_mean) / quantity_sd
  df$last_quantity2 <- (df$last_quantity2 - quantity_mean) / quantity_sd
  df$last_quantity3 <- (df$last_quantity3 - quantity_mean) / quantity_sd
  df$yday_quantity <- (df$yday_quantity - quantity_mean) / quantity_sd
  df
}

fit_lm_aic <- function(df) {
  df <- df[df$date != "2021-01-01",] ## remove jan 1st, as it contains NAs
  m <- lm(log_price ~  yday_log_price*yday_quantity +
                       last_log_price1*last_quantity1 +
                       last_log_price2*last_quantity2 +
                       period + month +
                       weekday,
          data=df) ## full model
  step(m)
}

fit_lm_final <- function(df) {
  df <- df[df$date != "2021-01-01",] ## remove jan 1st, as it contains NAs
  m <- lm(log_price ~yday_log_price + last_log_price1 +
                     last_quantity1 +
                     last_log_price2 + last_quantity2 + period + month + weekday +
                     last_log_price2:last_quantity2,
                     data=df) ## full model
  n <- nrow(df)
  fits <- predict(m, newdata=df, type='response', se=T)
  log_price_pred <- fits$fit
  resids <- df$log_price - log_price_pred; y <- resids

  write.csv(resids, file="residuals/lm-final.csv")

  y <- resids
  pdf("figures/acf_residuals.pdf")
  acf(y)
  dev.off()

  resid.df <- data.frame(y=resids, index=1:(length(resids)))
  first_of_month_period_1 <- (df$day == 1) & (df$period == 1)
  xticks <- ifelse(first_of_month_period_1,
                   df$date %>% months,
                   "")

  plt_y <- ggplot(resid.df, aes(y=y,x=index)) +
           scale_x_continuous(breaks=which(xticks!=""), labels=xticks[xticks!=""]) +
           theme(axis.title.x=element_blank(),
                 axis.text.x=element_text(angle=30, size=rel(rel_amnt)),
                 axis.title.y=element_text(size=rel(rel_amnt)),
                 axis.text.y=element_text(size=rel(rel_amnt))) +
           geom_line()
  ggsave(file="figures/y_lm_final.pdf", plot=plt_y)

  kpss_test_result <- kpss.test(resids, null="Level")
  print("printing kpss result to file...")
  sink("./kpss-result/lm-final.txt")
  print(kpss_test_result)
  sink()
  m
}

fit_lm_simple <- function(df) {
  ## remove period, month and weekday factors
  m <- lm(log_price ~  period + month +
                       weekday,
           data=df
  )

  n <- nrow(df)
  fits <- predict(m, newdata=df, type='response', se=T)
  log_price_pred <- fits$fit
  resids <- df$log_price - log_price_pred
  stopifnot(length(resids) == 48*365)


  ## simple check to see if we removed the trend
  kpss_test_result <- kpss.test(resids, null="Level")
  sink("./kpss-result/lm-simple.txt")
  #writelines(kpss_test_result, "./kpss-result/lm-simple.txt")
  print(kpss_test_result)
  sink()
  stopifnot(kpss_test_result$p.value > 0.05)

  ## simple plot of residuals
  data <- data.frame(y=resids, index=1:(length(resids)))
  first_of_month_period_1 <- (df$day == 1) & (df$period == 1)
  xticks <- ifelse(first_of_month_period_1,
                   df$date %>% months,
                   "")
  plt_y <- ggplot(data, aes(y=y,x=index)) +
           scale_x_continuous(breaks=which(xticks!=""), labels=xticks[xticks!=""]) +
           geom_line(aes(y=y, x=index)) +
           theme(axis.title.x=element_blank(),
                 axis.text.x=element_text(angle=30,size=rel(rel_amnt)),
                 axis.text.y=element_text(size=rel(rel_amnt)),
                 axis.title.y=element_text(size=rel(rel_amnt)))
#, height=1500, width=1800, units="px"
  ggsave("figures/y_resid.pdf", plot=plt_y)


  #pdf("figures/lm-3.pdf")
  plot(log_price_pred, pch=20, main="log price pred vs ix")
  plot(df$log_price, pch=20, main="true price vs ix")
  #lines(fits$fit)
  lines(log_price_pred)
  plot(resids, pch=20, main="resids vs ix")
  plot(resids[1:(n-1)], resids[2:n], pch=20, xlab="X_{t-1}", ylab="X_t", main="resids_t vs resids_{t-1}")
  plot(resids[1:(n-2)], resids[3:n], pch=20, xlab="X_{t-2}", ylab="X_t", main="resids_t vs resids_{t-2}")
  #dev.off()



  u <- 2
  u_ixs <- (head(resids, -1) > u) | (tail(resids, -1) > u)
  u_ixs[u_ixs %>% is.na] <- F


  #pdf("figures/lm-3-cond.pdf")
  plot(resids[head(u_ixs, -1)], resids[tail(u_ixs, -1)], pch=20, xlab="X_{t-1}", ylab="X_{t}")
  plot(resids[head(u_ixs, -2)], resids[tail(u_ixs, -2)], pch=20, xlab="X_{t-2}", ylab="X_{t}")
  #dev.off()

  write.csv(resids, "residuals/lm-simple.csv")
  m

}

fit_21 <- function() {
  print(a_coef)
  df <- get_model_fitting_df()
  model <- fit_lm_aic(df)
}

## jan 22 prediction
jan_22_apply_model <- function() {
  print(a_coef)
  train_nice_df <- get_nice_df(); train_df <- get_features_df(train_nice_df)
  df <- read_csv("./data/elexon-clean-jan-22.csv")

  true_prices_jan_22 <- df$imbalancePriceAmountGBP
  write.csv(true_prices_jan_22, "residuals/jan-22-true-prices.csv")

  df <- get_nice_df_(df)

  # add dec 31 to 22 data set so that we have yday prices available
  # for jan 1st 22. but remove them again when calculating residuals
  dec_31_21 <- train_nice_df[train_nice_df$settlementDate == "2021-12-31",]
  df <- rbind(dec_31_21,df)
  df <- residualize_df(df, train_df)
  df <- df[df$date != "2021-12-31",]
  stopifnot(df$log_price %>% is.na %>% any %>% not)
  stopifnot(df%>% is.na %>% any %>% not)

  train_df <- normalize_features(train_df)
  train_df <- train_df[train_df$date != "2021-01-01",] ## remove jan 1st, as it contains NAs
  m <- lm(log_price ~yday_log_price + last_log_price1 +
                     last_quantity1 +
                     last_log_price2 + last_quantity2 + period + month + weekday +
                     last_log_price2:last_quantity2,
                     data=train_df) ## full model
  n <- nrow(df)
  fits <- predict(m, newdata=df, type='response', se=T)
  log_price_pred <- fits$fit
  resids <- df$log_price - log_price_pred; y <- resids
  stopifnot(length(resids) == 31*48)
  write.csv(resids, "residuals/jan-22.csv")
  write.csv(log_price_pred, "residuals/jan-22-pred.csv")
}
