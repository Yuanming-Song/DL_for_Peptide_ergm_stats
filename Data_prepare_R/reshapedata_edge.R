# Configuration parameters
train_ratio <- 0.8  # 80% for training by default
valid_ratio <- 0.1  # 10% for validation by default
test_ratio <- 0.1   # 10% for testing by default
random_seed <- 123  # Random seed for reproducibility

# Validate ratios
if(abs(train_ratio + valid_ratio + test_ratio - 1) > 1e-10) {
    stop("Error: ratios must sum to 1")
}

# Set random seed
set.seed(random_seed)

# Function to process data into required format
process_data <- function(data) {
    # Check if data has the expected structure
    if (is.null(data$sequence) || is.null(data$value)) {
        # For ddedge data which might have a different structure
        if (!is.null(data$ddedge_data)) {
            return(data.frame(
                Index = seq_len(nrow(data$ddedge_data)) - 1,
                Feature = data$ddedge_data$sequence,
                Label = sprintf('"%s"', data$ddedge_data$value)
            ))
        }
        stop("Data structure not recognized")
    }
    
    # Create dataframe with Index starting from 0
    df <- data.frame(
        Index = seq_len(nrow(data)) - 1,
        Feature = data$sequence,
        Label = sprintf('"%s"', data$value)  # Quote the value
    )
    return(df)
}

# Define directories
edge_dir <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide"

# Function to load and process an .rda file
load_and_process <- function(filepath) {
    load(filepath)  # loads the data object
    # Determine which object was loaded based on filename
    if (grepl("monomer", filepath)) {
        processed <- process_data(monomer_data)
    } else if (grepl("dimer", filepath)) {
        processed <- process_data(dimer_data)
    } else if (grepl("ddedge", filepath)) {
        colnames(ddedge_data)[2]<-"value"
        # For ddedge, pass the entire loaded environment
        processed <- process_data(ddedge_data)
    }
    return(processed)
}

# Process each type of data
data_types <- list(
    dimer = list(
        path = file.path(edge_dir, "ML_ddEdge_dimer/Sequential_Peptides_edges/peptide_dimer_value.rda"),
        outdir = file.path(edge_dir, "ML_ddEdge_dimer/Sequential_Peptides_edges"),
        prefix = "dimer"
    ),
    monomer = list(
        path = file.path(edge_dir, "ML_ddEdge_monomer/Sequential_Peptides_edges/peptide_monomer_value.rda"),
        outdir = file.path(edge_dir, "ML_ddEdge_monomer/Sequential_Peptides_edges"),
        prefix = "monomer"
    ),
    ddedge = list(
        path = file.path(edge_dir, "ML_ddEdge/Sequential_Peptides_edges/peptide_ddedge.rda"),
        outdir = file.path(edge_dir, "ML_ddEdge/Sequential_Peptides_edges"),
        prefix = "ddedge"
    )
)

for (type in names(data_types)) {
    # Load and process the data
    df_all <- load_and_process(data_types[[type]]$path)
    
    # Split the data into train, validation, and test sets
    n <- nrow(df_all)
    train_size <- floor(train_ratio * n)
    valid_size <- floor(valid_ratio * n)
    
    # Generate random indices
    all_indices <- seq_len(n)
    train_idx <- sample(all_indices, size = train_size)
    remaining <- setdiff(all_indices, train_idx)
    valid_idx <- sample(remaining, size = valid_size)
    test_idx <- setdiff(remaining, valid_idx)
    
    # Create the datasets
    train_df <- df_all[train_idx, ]
    valid_df <- df_all[valid_idx, ]
    test_df <- df_all[test_idx, ]
    
    # Reset indices to be sequential in each set
    train_df$Index <- seq_len(nrow(train_df)) - 1
    valid_df$Index <- seq_len(nrow(valid_df)) - 1
    test_df$Index <- seq_len(nrow(test_df)) - 1
    
    # Create output directory if it doesn't exist
    dir.create(data_types[[type]]$outdir, recursive = TRUE, showWarnings = FALSE)
    
    # Save the datasets as CSV files
    write.csv(train_df, 
              file.path(data_types[[type]]$outdir, paste0(data_types[[type]]$prefix, "_train_seqs.csv")), 
              row.names = FALSE, quote = FALSE)
    write.csv(valid_df, 
              file.path(data_types[[type]]$outdir, paste0(data_types[[type]]$prefix, "_valid_seqs.csv")), 
              row.names = FALSE, quote = FALSE)
    write.csv(test_df, 
              file.path(data_types[[type]]$outdir, paste0(data_types[[type]]$prefix, "_test_seqs.csv")), 
              row.names = FALSE, quote = FALSE)
    
    cat("Files for", type, "saved in:", data_types[[type]]$outdir, "\n")
}

cat("Processing complete!\n")
cat("Train/Valid/Test split:", train_ratio, "/", valid_ratio, "/", test_ratio, "\n")
cat("Random seed used:", random_seed, "\n") 