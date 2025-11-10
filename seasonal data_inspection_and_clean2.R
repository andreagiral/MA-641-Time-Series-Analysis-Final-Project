# ---------- data_inspection_and_clean.R ----------
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lubridate)
  library(ggplot2)
  library(tidyr)
  library(zoo)        # na.approx, na.locf
  library(tseries)    # adf.test
  library(forecast)   # msts, mstl, autoplot
})

# ---- Load & parse datetime ----
df <- readr::read_csv(
  "combined_hourly_energy2.csv",
  show_col_types = FALSE
)

# Parse Datetime as POSIXct if not already
if (!inherits(df$Datetime, "POSIXct")) {
  df <- df %>% mutate(Datetime = ymd_hms(Datetime, tz = "UTC", quiet = TRUE))
}
df <- df %>% arrange(Datetime)

# ---- Basic info ----
str(df)
print(head(df, 5))

dupes <- df %>% filter(duplicated(Datetime))
cat("Duplicated timestamps:", nrow(dupes), "\n")

start <- min(df$Datetime, na.rm = TRUE)
end   <- max(df$Datetime, na.rm = TRUE)
cat("Data range:", format(start, "%Y-%m-%d %H:%M:%S"),
    "→", format(end, "%Y-%m-%d %H:%M:%S"), "\n")
cat("Number of records:", nrow(df), "\n")

# ---- Filter to common window ----
df <- df %>%
  filter(Datetime >= ymd("2005-05-01"),
         Datetime <= ymd("2018-08-03")) %>%
  arrange(Datetime)

# ---- Remove duplicates ----
dup_count <- sum(duplicated(df$Datetime))
cat("Duplicated index timestamps before reindexing:", dup_count, "\n")
if (dup_count > 0) {
  df <- df[!duplicated(df$Datetime), ]
}

# ---- Reindex to perfect hourly grid ----
idx <- seq.POSIXt(from = min(df$Datetime), to = max(df$Datetime), by = "hour")
df <- df %>%
  right_join(tibble(Datetime = idx), by = "Datetime") %>%
  arrange(Datetime)

# ---- Missing hours ----
na_all <- df %>%
  select(-Datetime) %>%
  mutate(row_na_all = if_else(rowSums(!is.na(.)) == 0, TRUE, FALSE)) %>%
  pull(row_na_all)
cat("Missing hourly timestamps:", sum(na_all), "\n")

# ---- Missing summary ----
na_summary <- colSums(is.na(select(df, -Datetime)))
na_pct     <- round(100 * na_summary / nrow(df), 2)
missing_table <- tibble(
  Column = names(na_summary),
  Missing_Count = as.integer(na_summary),
  Missing_Pct = na_pct
) %>% arrange(desc(Missing_Count))
print(missing_table)

# ============================================================
#                  CLEANING SECTION
# ============================================================
df_clean <- df

# Identify nearly-complete columns (< 1% missing)
complete_cols <- names(df_clean)[names(df_clean) != "Datetime"]
complete_cols <- complete_cols[
  sapply(complete_cols, function(cn) mean(is.na(df_clean[[cn]])) < 0.01)
]

# Interpolate small gaps (<= 6) then forward-fill tiny runs (<= 3)
for (cn in complete_cols) {
  df_clean[[cn]] <- zoo::na.approx(df_clean[[cn]], maxgap = 6, na.rm = FALSE)
  df_clean[[cn]] <- zoo::na.locf(df_clean[[cn]], maxgap = 3, na.rm = FALSE)
}

# fill the single missing DOM value (short-run linear interp)
if ("DOM" %in% names(df_clean)) {
  df_clean$DOM <- zoo::na.approx(df_clean$DOM, maxgap = 6, na.rm = FALSE)
  df_clean$DOM <- zoo::na.locf(df_clean$DOM, na.rm = FALSE)
  df_clean$DOM <- zoo::na.locf(df_clean$DOM, fromLast = TRUE, na.rm = FALSE)
}

# Drop highly incomplete columns
drop_cols <- c("PJM_LOAD", "EKPC", "NI", "DEOK", "FE", "COMED")
drop_cols <- intersect(drop_cols, names(df_clean))
if (length(drop_cols) > 0) {
  df_clean <- df_clean %>% select(-all_of(drop_cols))
}

# Save cleaned dataset
#readr::write_csv(df_clean, "hourly_energy_cleandata2.csv")
cat("✅ Cleaned dataset saved successfully.\n")
cat("Remaining columns:", paste(names(df_clean), collapse = ", "), "\n")

