# Define rebinning bins
monomer_rebins <- c(round(exp(seq(0, 5, 1))), 300)
dimer_rebins <- unique(round(monomer_rebins / 2))
dimer_rebins <- c(1, dimer_rebins[which(dimer_rebins > 1)])

# Function to rebin data for all columns in obj
rebin_data <- function(data, bins) {
  
  rebinned <-  bins[-length(bins)]  # Bin labels
  bins[length(bins)]<-bins[length(bins)]+0.1
  for (col in 2:ncol(data)) {  # Iterate over all columns except the first
    result <- numeric(length(bins) - 1)
    for (i in 1:(length(bins) - 1)) {
      idx <- which(data[, 1] >= bins[i] & data[, 1] < bins[i + 1])
      result[i] <- sum(data[idx, col])
    }
    rebinned <- cbind(rebinned,result)
  }
  rebinned <- as.data.frame(rebinned)
  colnames(rebinned)[2:ncol(rebinned)] <- colnames(data)[2:ncol(data)]  # Retain column names
  colnames(rebinned)[1] <- colnames(data)[1]  # Retain the first column name
  return(rebinned)
}

# Function to reformat rebinned data into the required style
reformat_data <- function(data) {
  # Transpose the data to match the required format
  transposed <- as.data.frame(t(data[, -1]))
  colnames(transposed) <- data[, 1]  # Use bin labels as column names
  transposed$Feature <- rownames(transposed)  # Add Feature column
  
  # Remove underscores from feature names
  transposed$Feature <- gsub("_", "", transposed$Feature)
  
  # Add a Label column by concatenating all bin values
  transposed$Label <- apply(transposed[, -ncol(transposed)], 1, function(x) paste0('"', paste(x, collapse = ","), '"'))
  
  # Add an Index column starting from 0
  transposed <- cbind(Index = seq_len(nrow(transposed)) - 1, transposed[, c("Feature", "Label")])
  
  return(transposed)
}

# Function to load, rebin, and combine data
load_and_combine <- function(filepaths, state, bins) {
  combined_data <- data.frame()
  for (filepath in filepaths) {
    load(filepath)  # Load the .rda file
    obj <- get(paste0(state, "_sizehis"))
    
    rebinned_data <- rebin_data(obj, bins)
    combined_data <- if (nrow(combined_data) == 0) rebinned_data else cbind(combined_data, rebinned_data[, -1])
  }
  return(combined_data)
}

# Function to split, reformat, and save data
split_and_save <- function(data, state, outdir) {
  # Split the data into train (80%), validation (10%), and test (10%)
  set.seed(123)
  n <- ncol(data) - 1  # Exclude the first column (bins)
  train_idx <- sample(2:(n + 1), size = floor(0.8 * n))
  remaining <- setdiff(2:(n + 1), train_idx)
  valid_idx <- sample(remaining, size = floor(0.5 * length(remaining)))
  test_idx <- setdiff(remaining, valid_idx)
  
  train_data <- data[, c(1, train_idx)]
  valid_data <- data[, c(1, valid_idx)]
  test_data  <- data[, c(1, test_idx)]
  
  # Reformat the data to match the required style
  train_data <- reformat_data(train_data)
  valid_data <- reformat_data(valid_data)
  test_data  <- reformat_data(test_data)
  
  # Save the datasets as CSV files
  write.csv(train_data, file.path(outdir, paste0(state, "_train_rebinned.csv")), row.names = FALSE, quote = FALSE)
  write.csv(valid_data, file.path(outdir, paste0(state, "_valid_rebinned.csv")), row.names = FALSE, quote = FALSE)
  write.csv(test_data,  file.path(outdir, paste0(state, "_test_rebinned.csv")), row.names = FALSE, quote = FALSE)
}

# Define file paths and output directory
maindir <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/Csizedist/"
outdir <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/Sequential_Peptides_Rebin/"
dir.create(outdir, showWarnings = FALSE)

# File paths for dimer and monomer data
files <- list(
  dimer = c(
    file.path(maindir, "dimer_cdist.rda"),
    file.path(maindir, "dimer_cdist_tripeptide.rda"),
    file.path(maindir, "dimer_cdist_tetrapeptide.rda")
  ),
  monomer = c(
    file.path(maindir, "monomer_cdist.rda"),
    file.path(maindir, "monomer_cdist_tripeptide.rda"),
    file.path(maindir, "monomer_cdist_tetrapeptide.rda")
  )
)

# Process, combine, split, and save data for each state
for (state in names(files)) {
  bins <- if (state == "dimer") dimer_rebins else monomer_rebins
  combined_data <- load_and_combine(files[[state]], state, bins)
  

   invalid_cols <- colnames(combined_data)[-1][colSums(combined_data[, -1])<0.999999999 ] 
  # Remove invalid columns
  combined_data <- combined_data[, !(colnames(combined_data) %in% invalid_cols)]
  
  # Report invalid column names in a log file
  log_file <- file.path(outdir, paste0(state, "_missing.log"))
  writeLines(invalid_cols, log_file)
  split_and_save(combined_data, state, outdir)
}
