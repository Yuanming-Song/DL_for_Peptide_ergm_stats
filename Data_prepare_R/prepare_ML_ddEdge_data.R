# Set library path
.libPaths("/dfs9/tw/yuanmis1/R_libs/")

# Required packages
required_packages <- c("parallel", "doParallel", "reshape", "foreach")

# Load required packages
for(pkg in required_packages) {
    if(!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg)
    }
    library(pkg, character.only = TRUE)
}

# Register parallel backend
registerDoParallel(cores = detectCores() - 1)

# Create output directories if they don't exist
dimer_dir <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_ddEdge_dimer/Sequential_Peptides_edges"
monomer_dir <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/Sequential_Peptides_edges"
ddedge_dir <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_ddEdge_dimer/Sequential_Peptides_edges"

dir.create(dimer_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(monomer_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ddedge_dir, recursive = TRUE, showWarnings = FALSE)

# Function to process a chunk of sequences
process_chunk <- function(chunk, edge_data) {
    # Get data for this chunk of sequences
    chunk_data <- edge_data[edge_data$seqname %in% chunk, ]
    
    # Calculate mean values for each sequence and state
    edge_means <- aggregate(value ~ seqname + state, data = chunk_data, FUN = mean)
    
    # Reshape to wide format for difference calculation
    edge_wide <- reshape(edge_means, 
                        idvar = "seqname", 
                        timevar = "state", 
                        direction = "wide")
    
    # Calculate ddedge and normalize values
    edge_wide$ddedge <- (edge_wide$value.dimer - edge_wide$value.monomer) / 300
    
    # Remove underscores from sequence names
    sequences <- gsub("_", "", edge_wide$seqname)
    
    # Create three separate dataframes with normalized values
    list(
        monomer = data.frame(
            sequence = sequences,
            value = edge_wide$value.monomer/300
        ),
        dimer = data.frame(
            sequence = sequences,
            value = edge_wide$value.dimer/300
        ),
        ddedge = data.frame(
            sequence = sequences,
            ddedge = edge_wide$ddedge
        )
    )
}

# Function to process network stats data
process_network_stats <- function(data_path) {
    # Load the network stats data
    load(data_path)
    
    # Process monomer data - check if already combined
    if(is.data.frame(final_results[["monomer"]])) {
        combined_df_mon <- final_results[["monomer"]]
    } else {
        combined_df_mon <- do.call(rbind, final_results[["monomer"]])
    }
    combined_df_mon$state <- "monomer"
    
    # Process dimer data - check if already combined
    if(is.data.frame(final_results[["dimer"]])) {
        combined_df_dim <- final_results[["dimer"]]
    } else {
        combined_df_dim <- do.call(rbind, final_results[["dimer"]])
    }
    combined_df_dim$state <- "dimer"
    
    # Combine all data
    combined_df <- rbind(combined_df_dim, combined_df_mon)
    
    # Filter for edge statistics at last frame
    edge_data <- combined_df[combined_df$stat_name == "edges" & 
                            combined_df$frame == max(combined_df$frame), ]
    
    # Get unique sequences and split into chunks
    unique_seqs <- unique(edge_data$seqname)
    num_cores <- detectCores() - 1
    seq_chunks <- split(unique_seqs, cut(seq_along(unique_seqs), num_cores, labels = FALSE))
    
    # Process chunks in parallel
    results_list <- foreach(chunk = seq_chunks, .packages = c("reshape")) %dopar% {
        process_chunk(chunk, edge_data)
    }
    
    # Combine results from all chunks
    final_results <- list(
        monomer = do.call(rbind, lapply(results_list, function(x) x$monomer)),
        dimer = do.call(rbind, lapply(results_list, function(x) x$dimer)),
        ddedge = do.call(rbind, lapply(results_list, function(x) x$ddedge))
    )
    
    return(final_results)
}

# Process each peptide type
base_paths <- list(
    #dipeptide = "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/Edgelist/Dipeptide",
    tripeptide = "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/Edgelist/Tripeptide",
    tetrapeptide = "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/Edgelist/Tetrapeptide"
)

# Initialize combined results
combined_results <- list(monomer = NULL, dimer = NULL, ddedge = NULL)

# Process each peptide type
for(peptide_type in names(base_paths)) {
    # List all network stats files
    files <- list.files(base_paths[[peptide_type]], pattern = paste0(peptide_type, "_network_stats.*single_node\\.rda$"), full.names = TRUE)
    
    if(length(files) == 0) {
        warning(paste("No network stats files found for", peptide_type, "in", base_path))
        return(NULL)
    }
    
    # Process each file and combine results
    all_results <- list(monomer = NULL, dimer = NULL, ddedge = NULL)
    for(file in files) {
        cat("Processing file:", file, "\n")
        results <- process_network_stats(file)
        
        # Append results
        all_results$monomer <- rbind(all_results$monomer, results$monomer)
        all_results$dimer <- rbind(all_results$dimer, results$dimer)
        all_results$ddedge <- rbind(all_results$ddedge, results$ddedge)
    }
    
    results<-all_results
    if(!is.null(results)) {
        combined_results$monomer <- rbind(combined_results$monomer, results$monomer)
        combined_results$dimer <- rbind(combined_results$dimer, results$dimer)
        combined_results$ddedge <- rbind(combined_results$ddedge, results$ddedge)
    }
}

# Save combined results
monomer_data <- combined_results$monomer
dimer_data <- combined_results$dimer
ddedge_data <- combined_results$ddedge
ddedge_dir <- dimer_dir #temporary
save(monomer_data, file = file.path(monomer_dir, "peptide_monomer_value.rda"))
save(dimer_data, file = file.path(dimer_dir, "peptide_dimer_value.rda"))
save(ddedge_data, file = file.path(ddedge_dir, "peptide_ddedge.rda"))

cat("Processing complete! Files saved:\n",
    "Monomer values:", file.path(monomer_dir, "peptide_monomer_value.rda"), "\n",
    "Dimer values:", file.path(dimer_dir, "peptide_dimer_value.rda"), "\n",
    "ddEdge values:", file.path(ddedge_dir, "peptide_ddedge.rda"), "\n") 