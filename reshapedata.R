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
load("../Csizedist/dimer_cdist.rda")

# Process both datasets
df_dimer <- process_data(dimer_sizehis)
load("../Csizedist/dimer_cdist_tripeptide.rda")

df_tripeptide <- process_data(dimer_sizehis)

# Split data (80% train, 10% validation, 10% test)
set.seed(123)
split_indices <- sample(1:nrow(df_dimer), size = nrow(df_dimer) * 0.8)
train_dimer <- df_dimer[split_indices, ]
valid_test_dimer <- df_dimer[-split_indices, ]
split_indices <- sample(1:nrow(valid_test_dimer), size = nrow(valid_test_dimer) * 0.5)
valid_dimer <- valid_test_dimer[split_indices, ]
test_dimer <- valid_test_dimer[-split_indices, ]

split_indices <- sample(1:nrow(df_tripeptide), size = nrow(df_tripeptide) * 0.8)
train_tripeptide <- df_tripeptide[split_indices, ]
valid_test_tripeptide <- df_tripeptide[-split_indices, ]
split_indices <- sample(1:nrow(valid_test_tripeptide), size = nrow(valid_test_tripeptide) * 0.5)
valid_tripeptide <- valid_test_tripeptide[split_indices, ]
test_tripeptide <- valid_test_tripeptide[-split_indices, ]

# Save CSVs
write.csv(train_dimer, "Sequential_Peptides/train_seqs_dist.csv", row.names = FALSE, quote = FALSE)
write.csv(valid_dimer, "Sequential_Peptides/valid_seqs_dist.csv", row.names = FALSE, quote = FALSE)
write.csv(test_dimer, "Sequential_Peptides/test_seqs_dist.csv", row.names = FALSE, quote = FALSE)

write.table(train_tripeptide, "Sequential_Peptides/train_seqs_dist.csv", row.names = FALSE, col.names = FALSE, sep = ",", quote = FALSE, append = TRUE)
write.table(valid_tripeptide, "Sequential_Peptides/valid_seqs_dist.csv", row.names = FALSE, col.names = FALSE, sep = ",", quote = FALSE, append = TRUE)
write.table(test_tripeptide, "Sequential_Peptides/test_seqs_dist.csv", row.names = FALSE, col.names = FALSE, sep = ",", quote = FALSE, append = TRUE)