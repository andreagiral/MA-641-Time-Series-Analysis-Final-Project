# ============================================================
#                  LIBRARIES
# ============================================================
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lubridate)
  library(ggplot2)
  library(tseries)
  library(forecast)
  library(tidyr)
  library(tibble)
  library(rugarch)
  library(FinTS)
})

# ============================================================
#              DATA SECTION DELIVERABLES
# ============================================================

df2 <- readr::read_csv("hourly_energy_cleandata2.csv", show_col_types = FALSE) %>%
  mutate(Datetime = ymd_hms(Datetime, tz = "UTC", quiet = TRUE)) %>%
  arrange(Datetime)

# ---- Summary statistics ----
desc <- df2 %>%
  tidyr::pivot_longer(-Datetime, names_to = "variable", values_to = "value") %>%
  dplyr::group_by(variable) %>%
  dplyr::summarise(
    mean = mean(value, na.rm = TRUE),
    sd   = sd(value,   na.rm = TRUE),
    min  = min(value,  na.rm = TRUE),
    max  = max(value,  na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::mutate(across(c(mean, sd, min, max), ~round(.x, 2)))
print(desc)

# ---- NA counts ----
print(colSums(is.na(select(df2, -Datetime))))

# ---- Target series ----
target_col <- "PJM_HOURLY_EST"
stopifnot(target_col %in% names(df2))

y <- df2 %>%
  select(Datetime, all_of(target_col)) %>%
  drop_na()

# ---- Visualization ----
# 1) Long-term trend
ggplot(y, aes(Datetime, .data[[target_col]])) +
  geom_line() +
  labs(title = "PJM Total Estimated Hourly Load (2005–2018)",
       x = "Time", y = "Energy Consumption (MW)") +
  theme_minimal()

# 2) Faceted yearly view
y %>%
  filter(year(Datetime) >= 2005 & year(Datetime) <= 2018) %>%
  mutate(Year = factor(year(Datetime))) %>%
  ggplot(aes(Datetime, .data[[target_col]])) +
  geom_line(color = "#3366CC", linewidth = 0.4) +
  facet_wrap(~ Year, ncol = 3, scales = "free_x") +
  labs(
    title = "PJM Hourly Load by Year (2005–2018)",
    x = "Time (within year)",
    y = "Energy Consumption (MW)"
  ) +
  theme_minimal()

# 3) Weekly pattern (mean by day of week)
dow <- y %>%
  mutate(dow = wday(Datetime, week_start = 1, label = TRUE, abbr = TRUE)) %>%
  group_by(dow) %>%
  summarise(avg = mean(.data[[target_col]], na.rm = TRUE), .groups = "drop")

ggplot(dow, aes(dow, avg)) +
  geom_col() +
  labs(title = "Average PJM Total Load by Day of Week",
       x = "Day of Week (Mon–Sun)",
       y = "Energy Consumption (MW)") +
  theme_minimal()

# 4) Daily pattern (mean by hour)
hod <- y %>%
  mutate(h = hour(Datetime)) %>%
  group_by(h) %>%
  summarise(avg = mean(.data[[target_col]], na.rm = TRUE), .groups = "drop")

ggplot(hod, aes(h, avg)) +
  geom_line() +
  labs(title = "Average PJM Total Load by Hour of Day",
       x = "Hour of the Day (0–23)",
       y = "Energy Consumption (MW)") +
  theme_minimal()

# ============================================================
#      BOX–JENKINS MODELING WORKFLOW (STEPS 1–6)
# ============================================================

# ---------- Common objects ----------
y_vec   <- as.numeric(y[[target_col]])
y_ts_24 <- ts(y_vec, frequency = 24)

n       <- length(y_ts_24)
h       <- 168           # 1-week-ahead horizon
n_train <- n - h

train_vec <- y_vec[1:n_train]
test_vec  <- y_vec[(n_train + 1):n]

train_ts_24  <- ts(train_vec, frequency = 24)
train_ts_168 <- ts(train_vec, frequency = 168)
train_msts   <- msts(train_vec, seasonal.periods = c(24, 168))

# ============================================================
# STEP 1 — STATIONARITY CHECKS (TS plot, ACF/PACF, ADF, KPSS)
# ============================================================
# ---------------------------
# 1A. Time Series Plot
# ---------------------------
autoplot(y_ts_24) +
  labs(title = "PJM Hourly Load (Full Series)",
       x = "Time", y = "MW") +
  theme_minimal()

#---------------------------
# 1B. ACF & PACF (Original) — subsampled
#---------------------------
# Subsample every 10th point (keeps structure but removes noise)
y_small <- ts(y_ts_24[seq(1, length(y_ts_24), by = 10)], frequency = 24)

lag_max <- 7 * 24  # 168 lags = 1 week

p_acf_raw <- ggAcf(y_small, lag.max = lag_max) +
  scale_x_continuous(breaks = seq(0, lag_max, by = 24),
                     labels = paste0(seq(0, 7), "d")) +
  labs(title = "ACF of PJM Hourly Load (Subsampled for Clarity)",
       x = "Lag (days)", y = "ACF") +
  theme_minimal()

p_pacf_raw <- ggPacf(y_small, lag.max = lag_max) +
  scale_x_continuous(breaks = seq(0, lag_max, by = 24),
                     labels = paste0(seq(0, 7), "d")) +
  labs(title = "PACF of PJM Hourly Load (Subsampled for Clarity)",
       x = "Lag (days)", y = "Partial ACF") +
  theme_minimal()

p_acf_raw
p_pacf_raw


# ---------------------------
# 1C. ADF + KPSS (Original)
# ---------------------------

verdict <- function(p, test = c("ADF","KPSS"), alpha = 0.05) {
  test <- match.arg(test)
  if (test == "ADF") {
    ifelse(p < alpha, "Stationary (reject unit root)", 
           "Non-stationary (fail to reject unit root)")
  } else {
    ifelse(p < alpha, "Non-stationary (reject stationarity)",
           "Stationary (fail to reject non-stationarity)")
  }
}

safe_kpss <- function(x, null = c("Level","Trend")) {
  null <- match.arg(null)
  tseries::kpss.test(na.omit(as.numeric(x)), null = null, lshort = TRUE)
}

adf_lvl    <- tseries::adf.test(as.numeric(y_ts_24))
kpss_lvl_L <- safe_kpss(y_ts_24, null = "Level")
kpss_lvl_T <- safe_kpss(y_ts_24, null = "Trend")

cat("\n===== ORIGINAL SERIES =====\n")
cat(sprintf("ADF p=%.4g → %s\n",       adf_lvl$p.value,    verdict(adf_lvl$p.value,"ADF")))
cat(sprintf("KPSS(Level) p=%.4g → %s\n", kpss_lvl_L$p.value, verdict(kpss_lvl_L$p.value,"KPSS")))
cat(sprintf("KPSS(Trend) p=%.4g → %s\n\n", kpss_lvl_T$p.value, verdict(kpss_lvl_T$p.value,"KPSS")))

# ---------------------------
# 1D. Seasonal differencing (lag = 24) subsampled
# ----------------------------
y_sdiff24 <- diff(y_ts_24, lag = 24)
y_sdiff24_small <- y_sdiff24[seq(1, length(y_sdiff24), by = 10)]

p_acf_sdiff <- ggAcf(y_sdiff24_small, lag.max = lag_max) +
  scale_x_continuous(breaks = seq(0, lag_max, by = 24),
                     labels = paste0(seq(0, 7), "d")) +
  labs(title = "ACF of Seasonal Diff (lag 24) — Subsampled",
       x = "Lag (days)", y = "ACF") +
  theme_minimal()

p_pacf_sdiff <- ggPacf(y_sdiff24_small, lag.max = lag_max) +
  scale_x_continuous(breaks = seq(0, lag_max, by = 24),
                     labels = paste0(seq(0, 7), "d")) +
  labs(title = "PACF of Seasonal Diff (lag 24) — Subsampled",
       x = "Lag (days)", y = "Partial ACF") +
  theme_minimal()

p_acf_sdiff
p_pacf_sdiff

adf_sdiff24 <- tseries::adf.test(na.omit(as.numeric(y_sdiff24)))
kpss_s_L    <- safe_kpss(y_sdiff24, null = "Level")
kpss_s_T    <- safe_kpss(y_sdiff24, null = "Trend")

cat("\n===== SEASONAL DIFFERENCE (lag 24) =====\n")
cat(sprintf("ADF p=%.4g → %s\n",       adf_sdiff24$p.value, verdict(adf_sdiff24$p.value,"ADF")))
cat(sprintf("KPSS(Level) p=%.4g → %s\n", kpss_s_L$p.value,   verdict(kpss_s_L$p.value,"KPSS")))
cat(sprintf("KPSS(Trend) p=%.4g → %s\n\n", kpss_s_T$p.value, verdict(kpss_s_T$p.value,"KPSS")))

# ============================================================
# STEP 1B — MULTI-SEASONAL DECOMPOSITION (MSTL)
# ============================================================

y_msts   <- msts(y_vec, seasonal.periods = c(24, 168))
fit_mstl <- mstl(y_msts)

# Full MSTL decomposition
autoplot(fit_mstl) +
  ggtitle("MSTL Decomposition — PJM Hourly Load")

# Extract components
cm <- as_tibble(as.data.frame(fit_mstl))
components <- bind_cols(tibble(Datetime = y$Datetime), cm) %>%
  rename(`Seasonal-24` = Seasonal24,
         `Seasonal-168` = Seasonal168)

# Daily seasonality
ggplot(components, aes(Datetime, `Seasonal-24`)) +
  geom_line() +
  scale_x_datetime(date_breaks = "1 year", date_labels = "%Y") +
  labs(title = "Daily Seasonality (s = 24)",
       x = "Year", y = "Seasonal Component") +
  theme_minimal()

# Weekly seasonality
ggplot(components, aes(Datetime, `Seasonal-168`)) +
  geom_line() +
  scale_x_datetime(date_breaks = "1 year", date_labels = "%Y") +
  labs(title = "Weekly Seasonality (s = 168)",
       x = "Year", y = "Seasonal Component") +
  theme_minimal()


# ============================================================
# STEP 2: FINDING MODELS (IDENTIFICATION)
# ============================================================
# Based on the ACF and PACF of the training series:
# - Strong daily seasonality at lag 24 → need D=1 and seasonal AR/MA terms.
# - Significant spikes at lags 1–2 in both ACF and PACF → AR(2) and MA(2) components.
# - Weekly seasonal spike at lag 168 → possible weekly differencing (D=1 at period 168).
#
# Candidate models justified:
# 1. ARIMA(2,0,2): baseline nonseasonal structure (AR(2) + MA(2)).
# 2. SARIMA(2,0,2)(1,1,1)[24]: captures daily seasonality indicated by ACF/PACF.
# 3. ARIMA(2,0,2)(0,1,0)[168]: weekly differencing matches 168-hour cycle.
# 4. auto.arima: data-driven model check.
# 5. ARIMA + Fourier(24,168): multi-seasonal alternative when SARIMA becomes heavy.
#
# No code is required here — this section is conceptual model identification.

# ============================================================
# STEP 3: PARAMETER ESTIMATION (FIT CANDIDATE MODELS)
# ============================================================

## (1) Nonseasonal ARIMA(2,0,2) -- ARMA-like baseline
fit_arima_202 <- Arima(
  train_ts_24,
  order  = c(2, 0, 2),
  method = "CSS-ML"
)

## (2) Daily SARIMA(2,0,2)(1,1,1)[24]
fit_sarima_day <- Arima(
  train_ts_24,
  order    = c(2, 0, 2),
  seasonal = list(order = c(1, 1, 1), period = 24),
  method   = "CSS-ML",
  optim.control = list(maxit = 200)
)

## (3) Weekly differenced ARIMA(2,0,2)(0,1,0)[168]
fit_week_sdiff_arma <- Arima(
  train_ts_168,
  order    = c(2, 0, 2),
  seasonal = list(order = c(0, 1, 0), period = 168),
  method   = "CSS-ML",
  optim.control = list(maxit = 200)
)

## (4) Constrained auto.arima (single-seasonal)
fit_auto <- auto.arima(
  train_ts_24,
  seasonal      = TRUE,
  stepwise      = TRUE,
  approximation = TRUE,
  max.order     = 8,
  max.p = 3, max.q = 3,
  max.P = 1, max.Q = 1,
  d = 0,              # level looks stationary by ADF
  D = 1,              # clear daily seasonality
  seasonal.test  = "ocsb",
  test           = "adf",
  allowdrift     = TRUE,
  allowmean      = TRUE
)

## (5) ARIMA (nonseasonal) + Fourier seasonal terms for 24 & 168
K <- c(5, 3)                        # harmonics for 24h & 168h
X_train <- fourier(train_msts, K = K)
stopifnot(nrow(X_train) == length(train_vec))

fit_fourier <- auto.arima(
  train_msts,
  seasonal      = FALSE,           # seasonality handled by Fourier regressors
  xreg          = X_train,
  d             = 0,
  allowmean     = FALSE,
  allowdrift    = FALSE,
  stepwise      = TRUE,
  approximation = TRUE,
  max.order     = 8,
  max.p = 4, max.q = 4
)

# ============================================================
# STEP 4: PARAMETER REDUNDANCY / MODEL COMPARISON
# ============================================================

cat("\n=== PARAMETER SUMMARIES (5 MODELS) ===\n")
print(summary(fit_arima_202))
print(summary(fit_sarima_day))
print(summary(fit_week_sdiff_arma))
print(summary(fit_auto))
print(summary(fit_fourier))

cat("\n=== INFORMATION CRITERIA (AIC & BIC ONLY) ===\n")

models <- list(
  "ARIMA(2,0,2)"                  = fit_arima_202,
  "SARIMA(2,0,2)(1,1,1)[24]"      = fit_sarima_day,
  "Week ARIMA(2,0,2)(0,1,0)[168]" = fit_week_sdiff_arma,
  "auto.arima (constrained)"      = fit_auto,
  "ARIMA + Fourier(24,168)"       = fit_fourier
)

get_aic <- function(m) {
  val <- suppressWarnings(tryCatch(AIC(m), error = function(e) NA_real_))
  as.numeric(val[1])
}

get_bic <- function(m) {
  val <- suppressWarnings(tryCatch(BIC(m), error = function(e) NA_real_))
  as.numeric(val[1])
}

aic_tbl <- tibble(
  Model = names(models),
  AIC   = sapply(models, get_aic),
  BIC   = sapply(models, get_bic)
) %>%
  arrange(AIC)

print(aic_tbl)

# ============================================================
# STEP 5: RESIDUAL ANALYSIS
# ============================================================

cat("\n=== RESIDUAL CHECKS (Ljung–Box + plots) ===\n")
checkresiduals(fit_arima_202)
checkresiduals(fit_sarima_day)
checkresiduals(fit_week_sdiff_arma)
checkresiduals(fit_auto)
checkresiduals(fit_fourier)

# (ACF of residuals, histogram, QQ plot, Ljung–Box output – these match
#  the flowchart box: ACF plot, histogram, QQ plot, Shapiro-Wilk, Ljung–Box.)

# ============================================================
# STEP 6: PREDICTION / FORECASTING
# ============================================================

# 1-week / 168-hour holdout forecasts
fc_arima_202        <- forecast(fit_arima_202,       h = h)
fc_sarima_day       <- forecast(fit_sarima_day,      h = h)
fc_week_sdiff_arma  <- forecast(fit_week_sdiff_arma, h = h)
fc_auto             <- forecast(fit_auto,            h = h)

# Fourier: need matching future regressors
X_future   <- fourier(train_msts, K = K, h = h)
fc_fourier <- forecast(fit_fourier, xreg = X_future, h = h)

acc_tbl <- bind_rows(
  as_tibble(accuracy(fc_arima_202,       test_vec)) %>% mutate(Model = "ARIMA(2,0,2)"),
  as_tibble(accuracy(fc_sarima_day,      test_vec)) %>% mutate(Model = "SARIMA(2,0,2)(1,1,1)[24]"),
  as_tibble(accuracy(fc_week_sdiff_arma, test_vec)) %>% mutate(Model = "Week ARIMA(2,0,2)(0,1,0)[168]"),
  as_tibble(accuracy(fc_auto,            test_vec)) %>% mutate(Model = "auto.arima (constrained)"),
  as_tibble(accuracy(fc_fourier,         test_vec)) %>% mutate(Model = "ARIMA + Fourier(24,168)")
) %>%
  select(Model, ME, RMSE, MAE, MAPE, MASE, ACF1) %>%
  arrange(RMSE)

cat("\n=== OUT-OF-SAMPLE ACCURACY (h = 168) ===\n")
print(acc_tbl)

# choose final model from simpler candidates
final_candidates <- acc_tbl %>%
  filter(Model %in% c("ARIMA(2,0,2)",
                      "SARIMA(2,0,2)(1,1,1)[24]",
                      "auto.arima (constrained)"))

best_model_name <- final_candidates$Model[1]
cat("\nSelected final model (by RMSE among simple candidates): ",
    best_model_name, "\n", sep = "")

# Refit best model on FULL series and forecast next week
y_ts_full <- ts(y_vec, frequency = 24)

fit_best_full <- switch(
  best_model_name,
  "ARIMA(2,0,2)" = Arima(
    y_ts_full, order = c(2,0,2), method = "CSS-ML"
  ),
  "SARIMA(2,0,2)(1,1,1)[24]" = Arima(
    y_ts_full,
    order    = c(2,0,2),
    seasonal = list(order = c(1,1,1), period = 24),
    method   = "CSS-ML",
    optim.control = list(maxit = 200)
  ),
  "auto.arima (constrained)" = auto.arima(
    y_ts_full,
    seasonal      = TRUE,
    stepwise      = TRUE,
    approximation = TRUE,
    max.order     = 8,
    max.p = 3, max.q = 3,
    max.P = 1, max.Q = 1,
    d = 0,
    D = 1
  )
)

fc_best <- forecast(fit_best_full, h = h)

autoplot(fc_best) +
  labs(
    title = paste0("Next-Week Forecast (", best_model_name, ") – ", target_col),
    x = "Time",
    y = "Energy Consumption (MW)"
  ) +
  theme_minimal()

checkresiduals(fit_best_full)

# ============================================================
# STEP 7: GARCH DIAGNOSTICS FOR PJM LOAD (SHOWING IT'S NOT NEEDED)
# ============================================================
# ------------------------------------------------------------
# 7A. Extract residuals from FINAL Box–Jenkins model
# ------------------------------------------------------------
res_full <- residuals(fit_best_full)
res_full <- as.numeric(na.omit(res_full))   # drop NAs from differencing, etc.

# Quick sanity check
summary(res_full)

# ------------------------------------------------------------
# 7B. ARCH LM test on residuals
#     H0: no ARCH effects (no time-varying variance)
# ------------------------------------------------------------
arch_test_24  <- FinTS::ArchTest(res_full, lags = 24)
arch_test_168 <- FinTS::ArchTest(res_full, lags = 168)

cat("\n=== ARCH LM TEST ON PJM RESIDUALS ===\n")
print(arch_test_24)
print(arch_test_168)

# In the report you’ll use the p-values here to argue:
# "We fail to reject H0 of no ARCH effects → variance is stable over time,
#  so a GARCH model is not necessary for PJM load."

# ------------------------------------------------------------
# 7C. Fit a simple GARCH(1,1) to residuals anyway
#     (to show that GARCH terms are small / not strongly significant)
# ------------------------------------------------------------

spec_pjm_garch <- ugarchspec(
  variance.model = list(
    model      = "sGARCH",
    garchOrder = c(1, 1)
  ),
  mean.model = list(
    armaOrder    = c(0, 0),      # residuals should already be mean-zero
    include.mean = FALSE
  ),
  distribution.model = "norm"    # could try "std" too
)

fit_pjm_garch <- ugarchfit(
  spec = spec_pjm_garch,
  data = res_full
)

cat("\n=== GARCH(1,1) FIT ON PJM RESIDUALS ===\n")
show(fit_pjm_garch)

# ------------------------------------------------------------
# 7D. Residual and volatility diagnostics
# ------------------------------------------------------------

# Standardized residuals (should look like white noise if GARCH captured everything)
pjm_std_resid <- residuals(fit_pjm_garch, standardize = TRUE)

par(mfrow = c(2, 2))
acf(pjm_std_resid,    main = "ACF of Std. Residuals (PJM GARCH)")
acf(pjm_std_resid^2,  main = "ACF of Squared Std. Residuals")
hist(pjm_std_resid,   main = "Histogram of Std. Residuals", xlab = "Std. resid")
qqnorm(pjm_std_resid); qqline(pjm_std_resid, col = "red")
par(mfrow = c(1, 1))

# Conditional volatility (sigma_t)
pjm_sigma <- sigma(fit_pjm_garch)

plot(
  pjm_sigma, type = "l",
  main = "Estimated Conditional Volatility – PJM Load Residuals",
  xlab = "Time index", ylab = "sigma_t"
)


