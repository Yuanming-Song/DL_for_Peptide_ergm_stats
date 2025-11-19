# Plot training metrics for ML_dEdge_gen generative model
# This script plots training curves including all losses vs. epoch

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

# MLflow output directory structure:
# out/{experiment_id}/{run_id}/metrics/{metric_name}
# Each metric file has format: timestamp value step

# Base path to MLflow output directory
# If base_path is not defined, set it to the default location
if (!exists("base_path")) {
  base_path <- "/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge_gen/v1+2"
}

mlflow_dir <- file.path(base_path, "out")

# Function to read MLflow metric file
read_mlflow_metric <- function(metric_path) {
  if (!file.exists(metric_path)) {
    return(NULL)
  }
  # Check if file has content
  if (file.info(metric_path)$size == 0) {
    return(NULL)
  }
  # MLflow metric format: timestamp value step
  tryCatch({
    data <- read.table(metric_path, header = FALSE, sep = " ", 
                       col.names = c("timestamp", "value", "step"),
                       stringsAsFactors = FALSE,
                       fill = TRUE)
    # Remove any rows with NA values
    data <- data[complete.cases(data), ]
    if (nrow(data) == 0) {
      return(NULL)
    }
    return(data)
  }, error = function(e) {
    warning(paste("Error reading", metric_path, ":", e$message))
    return(NULL)
  })
}

# Function to get run name from params
get_run_name <- function(run_dir) {
  run_name_file <- file.path(run_dir, "tags", "mlflow.runName")
  if (file.exists(run_name_file)) {
    return(readLines(run_name_file, n = 1))
  }
  # Fallback: use run ID
  return(basename(run_dir))
}

# Function to get model parameters from params directory
get_model_params <- function(run_dir) {
  params_dir <- file.path(run_dir, "params")
  if (!dir.exists(params_dir)) {
    return(list())
  }
  
  params <- list()
  param_files <- list.files(params_dir, full.names = TRUE)
  
  for (param_file in param_files) {
    param_name <- basename(param_file)
    param_value <- readLines(param_file, n = 1)
    # Try to convert to numeric if possible
    if (!is.na(suppressWarnings(as.numeric(param_value)))) {
      params[[param_name]] <- as.numeric(param_value)
    } else {
      params[[param_name]] <- param_value
    }
  }
  
  return(params)
}

# Function to create model label from parameters
create_model_label <- function(run_name, params, run_id) {
  # Try to extract key parameters from run_name or params
  if (length(params) > 0) {
    d_model <- ifelse("d_model" %in% names(params), params$d_model, "?")
    n_layers <- ifelse("n_layers" %in% names(params), params$n_layers, "?")
    n_heads <- ifelse("n_heads" %in% names(params), params$n_heads, "?")
    lr <- ifelse("lr" %in% names(params), params$lr, "?")
    
    # Create compact label with run ID suffix to ensure uniqueness
    # Use first 8 chars of run_id to keep it readable
    run_id_short <- substr(run_id, 1, 8)
    label <- paste0("d", d_model, "_l", n_layers, "_h", n_heads, "_lr", lr, "_", run_id_short)
    return(label)
  }
  # Fallback to run ID
  return(paste0("run_", substr(run_id, 1, 8)))
}

# Find all experiment directories
experiment_dirs <- list.dirs(mlflow_dir, recursive = FALSE, full.names = TRUE)

if (length(experiment_dirs) == 0) {
  stop(paste("No MLflow experiment directories found in:", mlflow_dir))
}

# Collect all runs from all experiments
all_runs <- list()
for (exp_dir in experiment_dirs) {
  run_dirs <- list.dirs(exp_dir, recursive = FALSE, full.names = TRUE)
  for (run_dir in run_dirs) {
    # Check if this is a valid run (has metrics directory)
    metrics_dir <- file.path(run_dir, "metrics")
    if (dir.exists(metrics_dir)) {
      # Check if train_loss file exists and has content
      train_loss_file <- file.path(metrics_dir, "train_loss")
      if (file.exists(train_loss_file) && file.info(train_loss_file)$size > 0) {
        all_runs[[length(all_runs) + 1]] <- run_dir
      }
    }
  }
}

if (length(all_runs) == 0) {
  stop(paste("No valid MLflow runs found in:", mlflow_dir))
}

# Sort runs by modification time of train_loss file (newest first for debugging)
run_info <- data.frame(
  run_dir = unlist(all_runs),
  stringsAsFactors = FALSE
)
run_info$mtime <- sapply(run_info$run_dir, function(rd) {
  train_loss_file <- file.path(rd, "metrics", "train_loss")
  if (file.exists(train_loss_file)) {
    return(file.info(train_loss_file)$mtime)
  }
  return(0)
})
run_info <- run_info[order(run_info$mtime, decreasing = TRUE), ]
all_runs <- run_info$run_dir

cat("Found", length(all_runs), "MLflow run(s).\n")
cat("Latest run:", basename(all_runs[1]), "\n")

# Collect metrics from all runs
all_metrics <- list()

# Helper function to align metrics by epoch/step (defined outside loop for efficiency)
align_metric <- function(metric_data, reference_steps) {
  if (is.null(metric_data) || nrow(metric_data) == 0) {
    return(rep(NA, length(reference_steps)))
  }
  # Merge by step to align with reference
  merged <- merge(data.frame(step = reference_steps), 
                  metric_data[, c("step", "value")], 
                  by = "step", all.x = TRUE, sort = TRUE)
  return(merged$value)
}

for (run_dir in all_runs) {
  tryCatch({
    metrics_dir <- file.path(run_dir, "metrics")
    
    # Read available metrics
    train_loss_file <- file.path(metrics_dir, "train_loss")
    train_dedge_file <- file.path(metrics_dir, "train_dedge_loss")
    train_recon_file <- file.path(metrics_dir, "train_recon_loss")
    valid_loss_file <- file.path(metrics_dir, "valid_loss")
    valid_dedge_file <- file.path(metrics_dir, "valid_dedge_loss")
    valid_recon_file <- file.path(metrics_dir, "valid_recon_loss")
    lr_file <- file.path(metrics_dir, "learning_rate")
    
    # Read metrics (handle missing files gracefully)
    train_loss_data <- read_mlflow_metric(train_loss_file)
    
    if (is.null(train_loss_data) || nrow(train_loss_data) == 0) {
      cat("Skipping run", basename(run_dir), "- no train_loss data\n")
      next  # Skip runs without train_loss
    }
    
    cat("Processing run:", basename(run_dir), "-", nrow(train_loss_data), "epochs\n")
    
    # Get model label (use run ID as fallback if label creation fails)
    run_id <- basename(run_dir)
    run_name <- tryCatch(get_run_name(run_dir), error = function(e) run_id)
    params <- tryCatch(get_model_params(run_dir), error = function(e) list())
    model_label <- tryCatch(create_model_label(run_name, params, run_id), error = function(e) {
      # Fallback: use run ID
      paste0("run_", substr(run_id, 1, 8))
    })
    
    # Read required metrics (train_loss, valid_loss, learning_rate)
    valid_loss_data <- read_mlflow_metric(valid_loss_file)
    lr_data <- read_mlflow_metric(lr_file)
    
    # Combine metrics into a data frame, aligning by epoch/step
    metrics_df <- data.frame(
      epoch = train_loss_data$step,
      train_loss = train_loss_data$value,
      valid_loss = align_metric(valid_loss_data, train_loss_data$step),
      learning_rate = align_metric(lr_data, train_loss_data$step),
      model = model_label,
      stringsAsFactors = FALSE
    )
    
    all_metrics[[length(all_metrics) + 1]] <- metrics_df
    cat("  Added", nrow(metrics_df), "data points for model:", model_label, "\n")
  }, error = function(e) {
    cat("ERROR processing run", basename(run_dir), ":", e$message, "\n")
    # Continue to next run instead of stopping
  })
}

cat("\nSuccessfully processed", length(all_metrics), "run(s).\n")

if (length(all_metrics) == 0) {
  stop("No valid metrics found in any runs.")
}

# Combine all metrics into single data frame
combined_metrics <- do.call(rbind, all_metrics)

# Get unique models
unique_models <- unique(combined_metrics$model)
cat("\nFound", length(unique_models), "unique model(s):", paste(unique_models, collapse = ", "), "\n")
cat("Total data points:", nrow(combined_metrics), "\n")
cat("Epoch range:", min(combined_metrics$epoch), "to", max(combined_metrics$epoch), "\n\n")

# Create color palette for models
n_models <- length(unique_models)
model_colors <- rainbow(n_models)
names(model_colors) <- unique_models

# Plot 1: Train Loss vs Epoch
p1 <- ggplot(combined_metrics, aes(x = epoch, y = train_loss, color = model)) +
  geom_line(size = 1, alpha = 0.8, na.rm = TRUE) +
  scale_color_manual(values = model_colors, name = "Model") +
  labs(
    title = "Training Loss vs Epoch",
    x = "Epoch",
    y = "Train Loss"
  ) +
  plttheme

print(ggplotly(p1))

# Plot 2: Valid Loss vs Epoch
if (any(!is.na(combined_metrics$valid_loss))) {
  plot_data <- combined_metrics[!is.na(combined_metrics$valid_loss), ]
  if (nrow(plot_data) > 0) {
    p2 <- ggplot(plot_data, aes(x = epoch, y = valid_loss, color = model)) +
      geom_line(size = 1, alpha = 0.8, na.rm = TRUE) +
      scale_color_manual(values = model_colors, name = "Model") +
      labs(
        title = "Validation Loss vs Epoch",
        x = "Epoch",
        y = "Valid Loss"
      ) +
      plttheme
    
    print(ggplotly(p2))
  }
}

# Plot 3: Learning Rate vs Epoch
if (any(!is.na(combined_metrics$learning_rate))) {
  plot_data <- combined_metrics[!is.na(combined_metrics$learning_rate), ]
  if (nrow(plot_data) > 0) {
    p3 <- ggplot(plot_data, aes(x = epoch, y = learning_rate, color = model)) +
      geom_line(size = 1, alpha = 0.8, na.rm = TRUE) +
      scale_color_manual(values = model_colors, name = "Model") +
      scale_y_log10() +
      labs(
        title = "Learning Rate vs Epoch",
        x = "Epoch",
        y = "Learning Rate (log scale)"
      ) +
      plttheme
    
    print(ggplotly(p3))
  }
}

# Plot 4-N: Combined plots for each individual model
# Create a plot for each model showing all metrics on one figure
cat("\nCreating combined plots for each model...\n")

# Define metrics to plot
plot_metrics <- c(
  "train_loss" = "Train Loss",
  "valid_loss" = "Valid Loss",
  "learning_rate" = "Learning Rate"
)

# Color palette for different metrics
metric_colors <- c(
  "Train Loss" = "#1f77b4",           # Blue
  "Valid Loss" = "#d62728",           # Red
  "Learning Rate" = "#2ca02c"           # Green
)

# Store combined plots for saving
combined_plots <- list()

# Create combined plot for each model
for (model_name in unique_models) {
  # Filter data for this model
  model_data <- combined_metrics[combined_metrics$model == model_name, ]
  
  # Reshape to long format for plotting
  plot_data_list <- list()
  
  for (metric in names(plot_metrics)) {
    if (any(!is.na(model_data[[metric]]))) {
      metric_data <- data.frame(
        epoch = model_data$epoch,
        value = model_data[[metric]],
        metric = plot_metrics[metric],
        stringsAsFactors = FALSE
      )
      # Remove NA rows
      metric_data <- metric_data[!is.na(metric_data$value), ]
      if (nrow(metric_data) > 0) {
        plot_data_list[[length(plot_data_list) + 1]] <- metric_data
      }
    }
  }
  
  if (length(plot_data_list) > 0) {
    plot_data_long <- do.call(rbind, plot_data_list)
    
    # Get available metrics for this model
    available_metrics <- unique(plot_data_long$metric)
    available_colors <- metric_colors[names(metric_colors) %in% available_metrics]
    
    # Check if learning rate is present
    has_lr <- "Learning Rate" %in% available_metrics
    
    if (has_lr) {
      # Separate loss metrics from learning rate
      loss_data <- plot_data_long[plot_data_long$metric != "Learning Rate", ]
      lr_data <- plot_data_long[plot_data_long$metric == "Learning Rate", ]
      
      # Calculate ranges for scaling
      loss_range <- range(loss_data$value, na.rm = TRUE)
      lr_range <- range(lr_data$value, na.rm = TRUE)
      
      # Scale learning rate to match loss range
      # Use a transformation to map LR to loss scale
      scale_factor <- (loss_range[2] - loss_range[1]) / (lr_range[2] - lr_range[1])
      lr_offset <- loss_range[1] - lr_range[1] * scale_factor
      
      # Create scaled LR data for plotting (but keep original for legend)
      lr_data_scaled <- lr_data
      lr_data_scaled$value <- lr_data$value * scale_factor + lr_offset
      lr_data_scaled$metric <- "Learning Rate"  # Keep original name for legend
      
      # Combine data
      plot_data_combined <- rbind(loss_data, lr_data_scaled)
      
      # Create the plot with dual y-axis
      p_combined <- ggplot(plot_data_combined, aes(x = epoch, y = value, color = metric)) +
        geom_line(size = 1, alpha = 0.8, na.rm = TRUE) +
        scale_color_manual(
          values = available_colors,
          name = "Metric"
        ) +
        scale_y_continuous(
          name = "Loss",
          sec.axis = sec_axis(
            ~ (. - lr_offset) / scale_factor,
            name = "Learning Rate",
            breaks = scales::pretty_breaks(n = 5)
          )
        ) +
        labs(
          title = paste("Training Metrics -", model_name),
          x = "Epoch",
          color = "Metric"
        ) +
        plttheme +
        theme(legend.position = "right")
      
    } else {
      # No learning rate, use simple plot
      p_combined <- ggplot(plot_data_long, aes(x = epoch, y = value, color = metric)) +
        geom_line(size = 1, alpha = 0.8, na.rm = TRUE) +
        scale_color_manual(
          values = available_colors,
          name = "Metric"
        ) +
        labs(
          title = paste("Training Metrics -", model_name),
          x = "Epoch",
          y = "Value",
          color = "Metric"
        ) +
        plttheme +
        theme(legend.position = "right")
    }
    
    print(ggplotly(p_combined))
    
    # Store plot for saving
    combined_plots[[model_name]] <- p_combined
  }
}

cat("Completed combined loss plots for", length(combined_plots), "model(s).\n")

# Save plots if requested
if (saveplt) {
  plot_dir <- file.path(base_path, "analysis", "plotting", "plots")
  dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Save individual metric plots
  ggsave(file.path(plot_dir, "train_loss.png"), p1, width = 12, height = 8, dpi = 300)
  if (exists("p2") && any(!is.na(combined_metrics$valid_loss))) {
    ggsave(file.path(plot_dir, "valid_loss.png"), p2, width = 12, height = 8, dpi = 300)
  }
  if (exists("p3") && any(!is.na(combined_metrics$learning_rate))) {
    ggsave(file.path(plot_dir, "learning_rate.png"), p3, width = 12, height = 8, dpi = 300)
  }
  
  # Save combined plots for each model
  for (model_name in names(combined_plots)) {
    # Create safe filename from model name
    safe_model_name <- gsub("[^A-Za-z0-9_]", "_", model_name)
    filename <- paste0("combined_losses_", safe_model_name, ".png")
    ggsave(file.path(plot_dir, filename), combined_plots[[model_name]], width = 14, height = 8, dpi = 300)
  }
  
  cat("Plots saved to:", plot_dir, "\n")
}

# ============================================================================
# Test Results Plots
# ============================================================================
cat("\nPlotting test results...\n")

# Source the test results plotting script
test_results_script <- file.path(base_path, "analysis", "plotting", "plot_mle_test_results.R")
if (file.exists(test_results_script)) {
  source(test_results_script)
} else {
  cat("Warning: Test results plotting script not found at:", test_results_script, "\n")
  cat("Skipping test results plots.\n")
}
