# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(gridExtra)
library(scales)

# Function to get the most recent log file
get_latest_log <- function(log_dir = "training_logs") {
  log_files <- list.files(log_dir, pattern = "training_log\\.csv$", full.names = TRUE)
  latest_log <- log_files[which.max(file.info(log_files)$mtime)]
  return(latest_log)
}

# Read the training log data
latest_log <- get_latest_log()
training_data <- read_csv(latest_log)

# Create a theme for consistent plotting
publication_theme <- theme_minimal() +
  theme(
    text = element_text(size = 12, family = "Arial"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 10),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = NA, color = "black")
  )

# 1. Loss Metrics Plot
loss_plot <- training_data %>%
  select(epoch, train_loss, valid_ce) %>%
  gather(key = "metric", value = "value", -epoch) %>%
  ggplot(aes(x = epoch, y = value, color = metric)) +
  geom_line(size = 1) +
  scale_color_manual(values = c("train_loss" = "#E69F00", 
                               "valid_ce" = "#56B4E9"),
                    labels = c("Training Loss", "Validation Cross-Entropy")) +
  labs(
    title = "Training Metrics",
    x = "Epoch",
    y = "Cross-Entropy Loss",
    color = "Metric"
  ) +
  publication_theme

# 2. Learning Rate Plot
lr_plot <- ggplot(training_data, aes(x = epoch, y = learning_rate)) +
  geom_line(color = "#CC79A7", size = 1) +
  scale_y_log10(labels = scales::scientific) +
  labs(
    title = "Learning Rate Schedule",
    x = "Epoch",
    y = "Learning Rate (log scale)"
  ) +
  publication_theme

# 3. Maximum Probability Error Plot
max_error_plot <- ggplot(training_data, aes(x = epoch, y = max_prob_error)) +
  geom_line(color = "#0072B2", size = 1) +
  labs(
    title = "Maximum Probability Error",
    x = "Epoch",
    y = "Max Error"
  ) +
  publication_theme

# 4. Early Stopping Progress
patience_plot <- ggplot(training_data, aes(x = epoch, y = patience_counter)) +
  geom_line(color = "#D55E00", size = 1) +
  labs(
    title = "Early Stopping Counter",
    x = "Epoch",
    y = "Patience Counter"
  ) +
  publication_theme

# Combine all plots
combined_plots <- gridExtra::grid.arrange(
  loss_plot, lr_plot, max_error_plot, patience_plot,
  ncol = 2
)

# Save plots
ggsave("training_plots.pdf", combined_plots, width = 12, height = 10)
ggsave("training_plots.png", combined_plots, width = 12, height = 10, dpi = 300)

# If you want to analyze test results as well
test_results_file <- list.files("results_seq_dist", 
                               pattern = "test_results\\.csv$", 
                               full.names = TRUE)[1]

if (!is.na(test_results_file)) {
  test_data <- read_csv(test_results_file)
  
  # Create prediction vs actual plot for distributions
  pred_vs_actual <- test_data %>%
    select(predict, label) %>%
    unnest(c(predict, label)) %>%
    ggplot(aes(x = label, y = predict)) +
    geom_point(alpha = 0.5, color = "#0072B2") +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(
      title = "Predicted vs Actual Probabilities",
      x = "Actual Probability",
      y = "Predicted Probability"
    ) +
    publication_theme
  
  # Save test results plots
  ggsave("test_results_plot.pdf", pred_vs_actual, width = 8, height = 6)
  ggsave("test_results_plot.png", pred_vs_actual, width = 8, height = 6, dpi = 300)
}

# Print summary statistics
cat("\nTraining Summary:\n")
cat("Total epochs:", max(training_data$epoch), "\n")
cat("Final Cross-Entropy:", training_data$valid_ce[nrow(training_data)], "\n")
cat("Best Cross-Entropy:", min(training_data$valid_ce), "\n")
cat("Final learning rate:", training_data$learning_rate[nrow(training_data)], "\n")
cat("Maximum probability error:", min(training_data$max_prob_error), "\n") 