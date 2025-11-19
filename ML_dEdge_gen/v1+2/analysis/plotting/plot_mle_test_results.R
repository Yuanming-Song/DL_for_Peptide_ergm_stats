# Plot MLE test results for generative model
# This script plots test results from MLE training, including:
# - Predicted dEdge vs True dEdge
# - dEdge error (MSE) vs True dEdge
# - Unique sequence fraction vs True dEdge
# - Generation statistics

# Load required libraries
library(ggplot2)
library(plotly)
library(dplyr)

# Assume plttheme is defined elsewhere (as per workspace rules)
# If not, we'll create a default theme
if (!exists("plttheme")) {
  plttheme <- theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10),
      legend.position = "bottom"
    )
}

# Set save flag (change to TRUE to save plots)
saveplt <- FALSE

# Base path to MLflow output directory
# If base_path is not defined, set it to the default location
if (!exists("base_path")) {
  base_path <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2"
}

out_dir <- file.path(base_path, "out")

# Function to find test result files
find_test_result_files <- function(out_dir) {
  test_files <- list()
  
  # Search in all experiment directories
  experiment_dirs <- list.dirs(out_dir, recursive = FALSE, full.names = TRUE)
  
  for (exp_dir in experiment_dirs) {
    run_dirs <- list.dirs(exp_dir, recursive = FALSE, full.names = TRUE)
    for (run_dir in run_dirs) {
      test_results_dir <- file.path(run_dir, "test_results")
      if (dir.exists(test_results_dir)) {
        test_csv_files <- list.files(test_results_dir, pattern = "test_results_.*\\.csv", full.names = TRUE)
        test_files <- c(test_files, test_csv_files)
      }
    }
  }
  
  return(test_files)
}

# Find all test result files
test_files <- find_test_result_files(out_dir)

if (length(test_files) == 0) {
  stop(paste("No test result files found in:", out_dir))
}

cat("Found", length(test_files), "test result file(s).\n")

# Collect test data from all files
all_test_data <- list()

for (test_file in test_files) {
  result <- tryCatch({
    # Read test CSV
    test_data <- read.csv(test_file, stringsAsFactors = FALSE)
    
    # Check if required columns exist
    required_cols <- c("true_dEdge", "avg_predicted_dEdge", "avg_dEdge_error", "unique_fraction")
    if (!all(required_cols %in% colnames(test_data))) {
      cat("Skipping", basename(test_file), "- missing required columns\n")
      NULL
    } else {
      # Extract run information from path
      # Path format: out/{experiment_id}/{run_id}/test_results/test_results_epoch_{epoch}.csv
      path_parts <- strsplit(test_file, "/")[[1]]
      run_id_idx <- which(path_parts == "out") + 2
      if (length(path_parts) > run_id_idx) {
        run_id <- path_parts[run_id_idx]
        experiment_id <- path_parts[run_id_idx - 1]
      } else {
        run_id <- "unknown"
        experiment_id <- "unknown"
      }
      
      # Extract epoch from filename
      filename <- basename(test_file)
      epoch <- gsub("test_results_epoch_|\\.csv", "", filename)
      
      # Add metadata
      test_data$run_id <- run_id
      test_data$experiment_id <- experiment_id
      test_data$epoch <- epoch
      test_data$source_file <- basename(test_file)
      
      cat("  Loaded test data from", basename(test_file), "-", nrow(test_data), "samples, run:", run_id, "\n")
      test_data
    }
  }, error = function(e) {
    cat("ERROR reading", basename(test_file), ":", e$message, "\n")
    NULL
  })
  
  if (!is.null(result)) {
    all_test_data[[length(all_test_data) + 1]] <- result
  }
}

if (length(all_test_data) == 0) {
  stop("No valid test data files could be loaded.")
}

# Combine all test data
combined_test_data <- do.call(rbind, all_test_data)

# Ensure numeric columns
combined_test_data$true_dEdge <- as.numeric(combined_test_data$true_dEdge)
combined_test_data$avg_predicted_dEdge <- as.numeric(combined_test_data$avg_predicted_dEdge)
combined_test_data$avg_dEdge_error <- as.numeric(combined_test_data$avg_dEdge_error)
combined_test_data$unique_fraction <- as.numeric(combined_test_data$unique_fraction)

# Remove any rows with NA values
combined_test_data <- combined_test_data[complete.cases(combined_test_data[, c("true_dEdge", "avg_predicted_dEdge", "avg_dEdge_error", "unique_fraction")]), ]

if (nrow(combined_test_data) == 0) {
  stop("No valid test data after filtering.")
}

cat("\nTotal test samples:", nrow(combined_test_data), "\n")
cat("Unique runs:", length(unique(combined_test_data$run_id)), "\n")
cat("True dEdge range:", min(combined_test_data$true_dEdge), "to", max(combined_test_data$true_dEdge), "\n")
cat("Average unique fraction:", mean(combined_test_data$unique_fraction), "\n")
cat("Average dEdge error (MSE):", mean(combined_test_data$avg_dEdge_error), "\n\n")

# Get unique runs for coloring
unique_runs <- unique(combined_test_data$run_id)
n_runs <- length(unique_runs)
run_colors <- rainbow(n_runs)
names(run_colors) <- unique_runs

# Initialize plot_dir for saving (if needed)
plot_dir <- file.path(base_path, "analysis", "plotting", "plots")
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# Plot 1: Predicted dEdge vs True dEdge (with identity line)
p1 <- ggplot(combined_test_data, aes(x = true_dEdge, y = avg_predicted_dEdge, color = run_id)) +
  geom_point(alpha = 0.6, size = 1.5) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red", size = 1) +
  scale_color_manual(values = run_colors, name = "Run") +
  labs(
    title = "Predicted dEdge vs True dEdge (Test Set)",
    x = "True dEdge",
    y = "Average Predicted dEdge"
  ) +
  plttheme

if (exists("saveplt") && saveplt) {
  ggsave(file.path(plot_dir, "test_predicted_vs_true_dedge.png"), p1, width = 12, height = 8, dpi = 300)
} else {
  print(ggplotly(p1))
}

# Plot 2: dEdge Error (MSE) vs True dEdge
p2 <- ggplot(combined_test_data, aes(x = true_dEdge, y = avg_dEdge_error, color = run_id)) +
  geom_point(alpha = 0.6, size = 1.5) +
  scale_color_manual(values = run_colors, name = "Run") +
  labs(
    title = "dEdge Error (MSE) vs True dEdge (Test Set)",
    x = "True dEdge",
    y = "Average dEdge Error (MSE)"
  ) +
  plttheme

if (exists("saveplt") && saveplt) {
  ggsave(file.path(plot_dir, "test_dedge_error_vs_true.png"), p2, width = 12, height = 8, dpi = 300)
} else {
  print(ggplotly(p2))
}

# Plot 3: Unique Sequence Fraction vs True dEdge
p3 <- ggplot(combined_test_data, aes(x = true_dEdge, y = unique_fraction, color = run_id)) +
  geom_point(alpha = 0.6, size = 1.5) +
  scale_color_manual(values = run_colors, name = "Run") +
  labs(
    title = "Unique Sequence Fraction vs True dEdge (Test Set)",
    x = "True dEdge",
    y = "Unique Sequence Fraction"
  ) +
  plttheme +
  ylim(0, 1)

if (exists("saveplt") && saveplt) {
  ggsave(file.path(plot_dir, "test_unique_fraction_vs_true.png"), p3, width = 12, height = 8, dpi = 300)
} else {
  print(ggplotly(p3))
}

# Plot 4: dEdge Error vs Sequence Length
if ("seq_length" %in% colnames(combined_test_data)) {
  combined_test_data$seq_length <- as.numeric(combined_test_data$seq_length)
  p4 <- ggplot(combined_test_data, aes(x = seq_length, y = avg_dEdge_error, color = run_id)) +
    geom_point(alpha = 0.6, size = 1.5) +
    scale_color_manual(values = run_colors, name = "Run") +
    labs(
      title = "dEdge Error vs Sequence Length (Test Set)",
      x = "Sequence Length",
      y = "Average dEdge Error (MSE)"
    ) +
    plttheme
  
  if (exists("saveplt") && saveplt) {
    ggsave(file.path(plot_dir, "test_dedge_error_vs_seq_length.png"), p4, width = 12, height = 8, dpi = 300)
  } else {
    print(ggplotly(p4))
  }
}

# Plot 5: Summary statistics by run
if (length(unique_runs) > 1) {
  summary_stats <- combined_test_data %>%
    group_by(run_id) %>%
    summarise(
      mean_unique_fraction = mean(unique_fraction, na.rm = TRUE),
      mean_dEdge_error = mean(avg_dEdge_error, na.rm = TRUE),
      median_dEdge_error = median(avg_dEdge_error, na.rm = TRUE),
      n_samples = n(),
      .groups = "drop"
    )
  
  cat("\nSummary Statistics by Run:\n")
  print(summary_stats)
  
  # Plot summary: Mean unique fraction by run
  p5a <- ggplot(summary_stats, aes(x = reorder(run_id, mean_unique_fraction), y = mean_unique_fraction)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
    labs(
      title = "Mean Unique Sequence Fraction by Run (Test Set)",
      x = "Run ID",
      y = "Mean Unique Fraction"
    ) +
    plttheme +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylim(0, 1)
  
  if (exists("saveplt") && saveplt) {
    ggsave(file.path(plot_dir, "test_summary_unique_fraction.png"), p5a, width = 12, height = 8, dpi = 300)
  } else {
    print(ggplotly(p5a))
  }
  
  # Plot summary: Mean dEdge error by run
  p5b <- ggplot(summary_stats, aes(x = reorder(run_id, mean_dEdge_error), y = mean_dEdge_error)) +
    geom_bar(stat = "identity", fill = "coral", alpha = 0.7) +
    labs(
      title = "Mean dEdge Error (MSE) by Run (Test Set)",
      x = "Run ID",
      y = "Mean dEdge Error (MSE)"
    ) +
    plttheme +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  if (exists("saveplt") && saveplt) {
    ggsave(file.path(plot_dir, "test_summary_dedge_error.png"), p5b, width = 12, height = 8, dpi = 300)
  } else {
    print(ggplotly(p5b))
  }
}

cat("\nCompleted plotting MLE test results.\n")

