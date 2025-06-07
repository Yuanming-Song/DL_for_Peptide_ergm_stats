# Load data

# Function to process data into required format
process_data <- function(data) {
  data <- data[, -1]  # Remove first column (not needed)
  colnames(data) <- gsub("_", "", colnames(data))  # Remove underscores in names
  
  # Convert to long format
  df <- as.data.frame(t(data))
  df$Feature <- rownames(df)
  df$Label <- apply(df[, -ncol(df)], 1, function(x) paste0('"', paste(x, collapse = ","), '"'))
  df <- df[, c("Feature", "Label")]
  df <- cbind(Index = seq_len(nrow(df)) - 1, df)  # Add index column starting from 0
  return(df)
}
# Assume process_data() is already defined and works as needed.
# Also assume maindir is defined as the main directory.
maindir <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/Csizedist/"
outdir<-"/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/Sequential_Peptides"
# Function to load and process an .rda file (assumes each .rda loads one object)
load_and_process <- function(filepath,state) {
  load(filepath)  # loads an object (e.g., dimer_sizehis or monomer_sizehis)
  obj <- get(paste0(state,"_sizehis"))
  processed <- process_data(obj)
  return(processed)
}

states <- c("dimer", "monomer")

for (state in states) {
  if (state == "dimer") {
    dipep_file    <- file.path(maindir, "dimer_cdist.rda")
    tripep_file   <- file.path(maindir, "dimer_cdist_tripeptide.rda")
    tetrapep_file <- file.path(maindir, "dimer_cdist_tetrapeptide.rda")
  } else if (state == "monomer") {
    dipep_file    <- file.path(maindir, "monomer_cdist.rda")
    tripep_file   <- file.path(maindir, "monomer_cdist_tripeptide.rda")
    tetrapep_file <- file.path(maindir, "monomer_cdist_tetrapeptide.rda")
  }
  
  # Load and process each dataset
  df_dipep    <- load_and_process(dipep_file,state)
  df_tripep   <- load_and_process(tripep_file,state)
  df_tetrapeep<- load_and_process(tetrapep_file,state)
  
  # Combine the data frames (assuming same columns)
  df_all <- rbind(df_dipep, df_tripep, df_tetrapeep)
  df_all[,1]<-seq_along(df_all[,1])
  # Split the combined data into train (80%), validation (10%), and test (10%)
  set.seed(123)
  n <- nrow(df_all)
  train_idx <- sample(seq_len(n), size = floor(0.8 * n))
  remaining <- setdiff(seq_len(n), train_idx)
  valid_idx <- sample(remaining, size = floor(0.5 * length(remaining)))
  test_idx <- setdiff(remaining, valid_idx)
  
  train_df <- df_all[train_idx, ]
  valid_df <- df_all[valid_idx, ]
  test_df  <- df_all[test_idx, ]
  
  # Save the datasets as CSV files under maindir
  if (state == "dimer") {
    write.csv(train_df, file.path(outdir, paste0(state, "_train_seqs_dist.csv")), row.names = FALSE, quote = FALSE)
    write.csv(valid_df, file.path(outdir, paste0(state, "_valid_seqs_dist.csv")), row.names = FALSE, quote = FALSE)
    write.csv(test_df,  file.path(outdir, paste0(state, "_test_seqs_dist.csv")), row.names = FALSE, quote = FALSE)
  } else if (state == "monomer") {
    write.csv(train_df, file.path(outdir, paste0(state, "_train_seqs_dist_monomer.csv")), row.names = FALSE, quote = FALSE)
    write.csv(valid_df, file.path(outdir, paste0(state, "_valid_seqs_dist_monomer.csv")), row.names = FALSE, quote = FALSE)
    write.csv(test_df,  file.path(outdir, paste0(state, "_test_seqs_dist_monomer.csv")), row.names = FALSE, quote = FALSE)
  }
}