# Iteration 2: Process only dEdge (tetrapeptides with C at pos 2,3,4)

# Load required packages
library(doParallel)
library(foreach)

# Configuration
train_ratio <- 0.8
valid_ratio <- 0.1
test_ratio <- 0.1
random_seed <- 123

if(abs(train_ratio + valid_ratio + test_ratio - 1) > 1e-10) {
    stop("Error: ratios must sum to 1")
}
set.seed(random_seed)

# Directory and file paths
base_dir <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge"
input_dir <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/Edgelist/Tetrapeptide"
output_dir <- file.path(base_dir, "data/iteration2/training/Sequential_Peptides_edges")

# Create output directory if it doesn't exist
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Load each .rda file separately (C2, C3, C4)
rda_files <- c(
    file.path(input_dir, "tetrapeptide_network_stats_C2_single_node.rda"),
    file.path(input_dir, "tetrapeptide_network_stats_C3_single_node.rda"),
    file.path(input_dir, "tetrapeptide_network_stats_C4_single_node.rda")
)

# Process each file and combine results
all_data <- NULL

# Set up parallel processing
num_cores <- detectCores() - 1  # Leave one core free
cl <- makeCluster(num_cores)
registerDoParallel(cl)

for (rda_file in rda_files) {
    load(rda_file)  # Loads final_results
    
    # Filter for edges in both monomer and dimer data
    monomer_edges <- final_results[[1]][final_results[[1]]$stat_name == "edges", ]
    dimer_edges <- final_results[[2]][final_results[[2]]$stat_name == "edges", ]
    
    # Get unique sequences
    unique_seqs <- unique(monomer_edges$seqname)
    
    # Process sequences in parallel
    results <- foreach(seq = unique_seqs, .combine = rbind) %dopar% {
        # Get data for this sequence
        monomer_seq_data <- monomer_edges[monomer_edges$seqname == seq, ]
        dimer_seq_data <- dimer_edges[dimer_edges$seqname == seq, ]
        
        # Get final frame data
        final_monomer <- monomer_seq_data[monomer_seq_data$frame == max(monomer_seq_data$frame), ]
        final_dimer <- dimer_seq_data[dimer_seq_data$frame == max(dimer_seq_data$frame), ]
        
        # Calculate dEdge value
        dedge_value <- (final_dimer$value - final_monomer$value) / 300
        
        # Return row for this sequence
        data.frame(
            sequence = seq,
            value = dedge_value,
            stringsAsFactors = FALSE
        )
    }
    
    # Combine with previous results
    if (is.null(all_data)) {
        all_data <- results
    } else {
        all_data <- rbind(all_data, results)
    }
}

# Stop parallel processing
stopCluster(cl)


filtered_data <- all_data

# Prepare DataFrame
df_all <- data.frame(
    Index = seq_len(nrow(filtered_data)) - 1,
    Feature = filtered_data$sequence,
    Label = sprintf('"%s"', filtered_data$value)
)

# Split into train/valid/test
n <- nrow(df_all)
train_size <- floor(train_ratio * n)
valid_size <- floor(valid_ratio * n)

all_indices <- seq_len(n)
train_idx <- sample(all_indices, size = train_size)
remaining <- setdiff(all_indices, train_idx)
valid_idx <- sample(remaining, size = valid_size)
test_idx <- setdiff(remaining, valid_idx)

train_df <- df_all[train_idx, ]
valid_df <- df_all[valid_idx, ]
test_df <- df_all[test_idx, ]

# Reset indices
train_df$Index <- seq_len(nrow(train_df)) - 1
valid_df$Index <- seq_len(nrow(valid_df)) - 1
test_df$Index <- seq_len(nrow(test_df)) - 1

# Save CSVs
write.csv(train_df, file.path(output_dir, "dedge_train_seqs.csv"), row.names = FALSE, quote = FALSE)
write.csv(valid_df, file.path(output_dir, "dedge_valid_seqs.csv"), row.names = FALSE, quote = FALSE)
write.csv(test_df, file.path(output_dir, "dedge_test_seqs.csv"), row.names = FALSE, quote = FALSE)

cat("Files for dEdge iteration 2 saved in:", output_dir, "\n")
cat("Train/Valid/Test split:", train_ratio, "/", valid_ratio, "/", test_ratio, "\n")
cat("Random seed used:", random_seed, "\n") 