# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Set save flag (change to TRUE to save plots)
save_plots <- FALSE

# Read the metrics data
metrics <- read.csv("/Users/song/Documents/Research/HPC/dfs2/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_ddEdge/results_transformer/training_metrics.csv")
test_results <- read.csv("/Users/song/Documents/Research/HPC/dfs2/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_ddEdge/results_transformer/test_results.csv")

# 1. MSE Learning Curves
p1 <- ggplot(metrics, aes(x = epoch)) +
  geom_line(aes(y = log10(train_mse), color = "Training"), size = 1) +
  geom_line(aes(y = log10(valid_mse), color = "Validation"), size = 1) +
  geom_point(data = subset(metrics, is_best == TRUE), 
             aes(y = valid_mse), color = "red", size = 3) +
  scale_color_manual(values = c("Training" = "#1f77b4", "Validation" = "#2ca02c")) +
  labs(title = "Mean Squared Error Learning Curves",
       x = "Epoch",
       y = "MSE",
       color = "Dataset") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    legend.position = "bottom"
  )+xlim(1,90)

# 2. MAE Learning Curves
p2 <- ggplot(metrics, aes(x = epoch)) +
  geom_line(aes(y = train_mae, color = "Training"), size = 1) +
  geom_line(aes(y = valid_mae, color = "Validation"), size = 1) +
  geom_point(data = subset(metrics, is_best == TRUE), 
             aes(y = valid_mae), color = "red", size = 3) +
  scale_color_manual(values = c("Training" = "#1f77b4", "Validation" = "#2ca02c")) +
  labs(title = "Mean Absolute Error Learning Curves",
       x = "Epoch",
       y = "MAE",
       color = "Dataset") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    legend.position = "bottom"
  )+xlim(0,50)+ylim(0,2)

# 3. True vs Predicted Values
p3 <- ggplot(test_results, aes(x = True_Value, y = Prediction)) +
  geom_point(alpha = 0.5, color = "#1f77b4") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "True vs Predicted Values",
       x = "True Value",
       y = "Predicted Value") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  ) +
  coord_fixed(ratio = 1)  # Makes the plot square with equal scales

# Arrange plots in a grid
p1_plotly <- ggplotly(p1)
p2_plotly <- ggplotly(p2) 
p3_plotly <- ggplotly(p3)

subplot(list(p1_plotly, p2_plotly, p3_plotly), nrows=2, ncol=2)

# Save plots if save_plots is TRUE
if(save_plots) {
  ggsave("results_transformer/learning_curves_combined.pdf", combined_plot, width = 15, height = 12)
  ggsave("results_transformer/learning_curves_combined.png", combined_plot, width = 15, height = 12, dpi = 300)
}

# Print summary statistics
cat("\nSummary Statistics:\n")
cat("Best Validation MSE:", min(metrics$valid_mse), "\n")
cat("At Epoch:", metrics$epoch[which.min(metrics$valid_mse)], "\n")
cat("Corresponding MAE:", metrics$valid_mae[which.min(metrics$valid_mse)], "\n")

cat("\nFinal Metrics:\n")
cat("Training MSE:", tail(metrics$train_mse, 1), "\n")
cat("Validation MSE:", tail(metrics$valid_mse, 1), "\n")
cat("Training MAE:", tail(metrics$train_mae, 1), "\n")
cat("Validation MAE:", tail(metrics$valid_mae, 1), "\n")

# Calculate and print test set metrics
test_mae <- mean(test_results$Absolute_Error)
test_mse <- mean((test_results$True_Value - test_results$Prediction)^2)
cat("\nTest Set Metrics:\n")
cat("Test MSE:", test_mse, "\n")
cat("Test MAE:", test_mae, "\n")