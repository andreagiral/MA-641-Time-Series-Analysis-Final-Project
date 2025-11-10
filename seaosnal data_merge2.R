# ---------- data_merge.R ----------
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(purrr)
})

# Set your folder path (adjust if needed)
path <- "C:/Users/andre/OneDrive/MSDS Masters/Fall 2025/MA641/Seasonal Data"
setwd(path)

# List of files (kept same names/order as Python)
files <- c(
  "AEP_hourly.csv", "COMED_hourly.csv", "DAYTON_hourly.csv",
  "DEOK_hourly.csv", "DOM_hourly.csv", "DUQ_hourly.csv",
  "EKPC_hourly.csv", "FE_hourly.csv", "NI_hourly.csv",
  "PJME_hourly.csv", "PJMW_hourly.csv", "pjm_hourly_est.csv",
  "PJM_Load_hourly.csv"
)

# Helper to read one file, pick the non-Datetime numeric column, and rename
read_and_rename <- function(f) {
  df <- readr::read_csv(f, show_col_types = FALSE)
  # Ensure Datetime exists and is parsed
  if (!"Datetime" %in% names(df)) stop(paste("No 'Datetime' in", f))
  # Identify the first non-Datetime column (mimics Python logic)
  mw_col <- setdiff(names(df), "Datetime")[1]
  var_name <- toupper(str_remove(str_remove(f, "_hourly\\.csv$"), "\\.csv$"))
  df %>%
    select(Datetime, all_of(mw_col)) %>%
    rename(!!var_name := all_of(mw_col))
}

dfs <- map(files, read_and_rename)

# Outer-join reduce on Datetime (like how='outer' merges in Python)
merged_df <- Reduce(function(x, y) full_join(x, y, by = "Datetime"), dfs) %>%
  arrange(Datetime)

# Save
readr::write_csv(merged_df, "combined_hourly_energy2.csv")
cat("✅ Combined dataset created successfully!\n")
cat("Shape:", paste0(nrow(merged_df), " x ", ncol(merged_df)), "\n")
cat("Columns:", paste(names(merged_df), collapse = ", "), "\n")
