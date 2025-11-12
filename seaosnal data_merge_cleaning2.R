# ---------- data_merge.R ----------
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(purrr)
  library(zoo)
  library(stringr)
  library(lubridate)
  library(tibble)
})

# Set your folder path (adjust if needed)
path <- "C:/Users/andre/OneDrive/MSDS Masters/Fall 2025/MA641"
setwd(path)
# ============================================================
#                  MERGING SECTION
# ============================================================
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
#readr::write_csv(merged_df, "combined_hourly_energy2.csv")
cat("Combined dataset created successfully!\n")
cat("Shape:", paste0(nrow(merged_df), " x ", ncol(merged_df)), "\n")
cat("Columns:", paste(names(merged_df), collapse = ", "), "\n")

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
cat("Cleaned dataset saved successfully.\n")
cat("Remaining columns:", paste(names(df_clean), collapse = ", "), "\n")


