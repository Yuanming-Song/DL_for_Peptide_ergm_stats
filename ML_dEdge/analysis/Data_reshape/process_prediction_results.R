#!/usr/bin/env Rscript

# Load required libraries
library(tidyverse)
library(data.table)

# Set paths
prediction_dir <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/data/iteration2/Prediction"
output_dir <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/analysis/Data_reshape"

# Function to extract length and position from filename
extract_info <- function(filename) {
    # Extract length and position using regex
    length_match <- str_extract(filename, "(?<=_)[0-9]+(?=mer)")
    pos_match <- str_extract(filename, "(?<=pos)[0-9]+")
    
    return(list(
        length = as.numeric(length_match),
        position = as.numeric(pos_match)
    ))
}

# Get all result files
result_files <- list.files(prediction_dir, pattern = "*_results.csv", full.names = TRUE)

# Initialize empty list to store processed data
processed_data <- list()

# Process each file
for (file in result_files) {
    # Read the CSV file
    data <- fread(file)
    
    # Extract information from filename
    info <- extract_info(basename(file))
    
    # Add metadata columns
    data <- data %>%
        mutate(
            length = info$length,
            position = info$position,
            dataset = "prediction"
        )
    
    # Add to processed data list
    processed_data[[basename(file)]] <- data
}

# Combine all data
combined_data <- rbindlist(processed_data)

# Create binned distribution data for plotting
dist_data <- combined_data %>%
    select(Prediction, length) %>%
    mutate(
        dataset = "prediction",
        length = factor(length)
    ) %>%
    # Create bins for dEdge values
    mutate(bin = cut(Prediction, 
                    breaks = seq(-1, 3, by = 0.1),
                    include.lowest = TRUE)) %>%
    # Count occurrences in each bin for each length
    group_by(length, bin) %>%
    summarise(count = n(), .groups = "drop") %>%
    # Normalize counts by length
    group_by(length) %>%
    mutate(normalized_count = count / sum(count)) %>%
    ungroup()

# Save only the distribution data
save(dist_data, file = file.path(output_dir, "prediction_results.rda"))

# Print summary
cat("Processed", length(result_files), "files\n")
cat("Total sequences:", nrow(combined_data), "\n")
cat("Unique lengths:", length(unique(combined_data$length)), "\n")
cat("Data saved to:", file.path(output_dir, "prediction_results.rda"), "\n") 