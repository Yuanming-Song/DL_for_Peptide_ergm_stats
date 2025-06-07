# Load required libraries
library(ggplot2)
library(plotly)
library(dplyr)
library(tidyr)
library(cowplot)

# Custom theme for text sizes
text_theme <- theme(
  plot.title = element_text(size = plot_title_size),
  axis.title = element_text(size = axis_title_size),
  axis.text = element_text(size = axis_text_size),
  legend.title = element_text(size = legend_title_size),
  legend.text = element_text(size = legend_text_size)
)

# Function to read iteration2 data
read_iteration2_data <- function(base_path) {
    data_path <- file.path(base_path, "data/iteration2/training/Sequential_Peptides_edges")
    results_path <- file.path(base_path, "results/iteration2")
    
    # Read all data files
    iter2_test_results <- read.csv(file.path(data_path, "curriculum_test_results_Transformer_lr_0.2_bs_1024.csv"))
    iter2_train_data <- read.csv(file.path(data_path, "dedge_train_seqs.csv"))
    iter2_valid_data <- read.csv(file.path(data_path, "dedge_valid_seqs.csv"))
    iter2_test_data <- read.csv(file.path(data_path, "dedge_test_seqs.csv"))
    
    # Prepare error data
    iter2_error_data <- iter2_test_results %>%
        mutate(abs_error = abs(Label - Prediction),
               squared_error = (Label - Prediction)^2) %>%
        select(Feature, Label, abs_error, squared_error) %>%
        gather(key = "error_type", value = "error", -c(Feature, Label))
    
    # Prepare distribution data with normalized frequencies
    iter2_dist_data <- bind_rows(
        data.frame(dataset = "Train", value = iter2_train_data$Label),
        data.frame(dataset = "Validation", value = iter2_valid_data$Label),
        data.frame(dataset = "Test", value = iter2_test_data$Label)
    ) %>%
    group_by(dataset) %>%
    mutate(value = value,
           density = n() / sum(n())) %>%
    ungroup()
    
    return(list(
        test_results = iter2_test_results,
        error_data = iter2_error_data,
        dist_data = iter2_dist_data
    ))
}

# Read data
if (readdata) {
  iter2_data <- read_iteration2_data("~/Documents/Research/HPC/dfs2/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/")
}

# Plot 1: Scatter plot with points colored by feature
p1 <- ggplot(iter2_data$test_results, aes(x = Label, y = Prediction, color = Feature)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
    labs(x = "True dEdge", y = "Predicted dEdge", 
         color = "Sequence") +
    plttheme +
    text_theme

# Plot 2: Distribution of dEdge values across different datasets
p2 <- ggplot(iter2_data$dist_data, aes(x = value, color = dataset)) +
    geom_density(alpha = 0.5, fill = NA) +
    labs(x = "dEdge", y = "Normalized Frequency",
         color = "") +
    plttheme +
    text_theme +
    theme(legend.direction = "vertical",
          legend.key.size = unit(8, "pt")) +
    guides(color = guide_legend(nrow = 3, ncol = 1))

# Plot 3: Error vs Label scatter plot with different shapes for error types
p3 <- ggplot(iter2_data$error_data, aes(x = Label, y = error, shape = error_type, color = error_type)) +
    geom_point() +
    scale_shape_manual(values = c("abs_error" = 24, "squared_error" = 22),
                      labels = c("Absolute Error", "Squared Error")) +
    scale_color_manual(values = c("abs_error" = "#2ecc71", "squared_error" = "#e74c3c"),
                      labels = c("Absolute Error", "Squared Error")) +
    labs(x = "True dEdge", y = "Error",
         shape = "",
         color = "") +
    ylim(0,1) +
    plttheme +
    text_theme +
    theme(legend.direction = "vertical",
          legend.position = "bottom",
          legend.key.size = unit(8, "pt")) +
    guides(shape = guide_legend(nrow = 2, ncol = 1),
           color = guide_legend(nrow = 2, ncol = 1))

# Combine plots in one row
combined_plot <- plot_grid(p1+theme(legend.position = "none"), 
                         p2+theme(legend.position = c(0.8,0.9)), 
                         p3+theme(legend.position = c(0.8,1)), 
                         ncol = 3,
                         align = 'h',
                         rel_widths = c(1, 1, 1))
combined_plot
# Display interactive plots
print(ggplotly(p1))
print(ggplotly(p2))
print(ggplotly(p3))

# Save high-resolution plots
if (exists("saveplt") && saveplt) {
  # Save combined plot
  save_plot(file.path(base_path_plt, "iteration2_combined_analysis.png"), 
            combined_plot,
            base_width = 6.5,
            base_height = 2,
            dpi = 1100,
            units = "in")
}

# Print analysis results
print("Top 10 predictions:")
print(iter2_data$test_results %>% 
        arrange(desc(Prediction)) %>% 
        head(10))

print("Bottom 10 predictions:")
print(iter2_data$test_results %>% 
        arrange(Prediction) %>% 
        head(10))

# Calculate and print summary statistics
print("Summary Statistics for Test Results:")
print(summary(iter2_data$test_results[c("Label", "Prediction")]))

# Calculate performance metrics
mse <- mean((iter2_data$test_results$Label - iter2_data$test_results$Prediction)^2)
mae <- mean(abs(iter2_data$test_results$Label - iter2_data$test_results$Prediction))
r2 <- 1 - sum((iter2_data$test_results$Label - iter2_data$test_results$Prediction)^2) / 
         sum((iter2_data$test_results$Label - mean(iter2_data$test_results$Label))^2)

print("Overall Model Performance Metrics:")
print(paste("MSE:", mse))
print(paste("MAE:", mae))
print(paste("R-squared:", r2))

# Generate LaTeX table for top and bottom 10
top_10 <- iter2_data$test_results %>% 
  arrange(desc(Prediction)) %>% 
  head(10) %>%
  select(Feature, Prediction)

bottom_10 <- iter2_data$test_results %>% 
  arrange(Prediction) %>% 
  head(10) %>%
  select(Feature, Prediction)

# Create LaTeX table
latex_table <- c(
  "\\begin{table}[H]",
  "    \\centering",
  "    \\caption{Top 10 and Bottom 10 sequences based on predicted dEdge values (Iteration 2).}",
  "    \\label{tab:top_bottom_10_ddedge_iter2}",
  "    \\begin{tabular}{|c|c||c|c|}",
  "        \\hline",
  "        \\multicolumn{2}{|c||}{Top 10} & \\multicolumn{2}{c|}{Bottom 10} \\\\",
  "        \\hline",
  "        Sequence & dEdge & Sequence & dEdge \\\\",
  "        \\hline"
)

# Combine rows
for(i in 1:10) {
  row <- sprintf("        %s & %.4f & %s & %.4f \\\\",
                 top_10$Feature[i], top_10$Prediction[i],
                 bottom_10$Feature[i], bottom_10$Prediction[i])
  latex_table <- c(latex_table, row)
}

# Add closing lines
latex_table <- c(latex_table,
                 "        \\hline",
                 "    \\end{tabular}",
                 "\\end{table}")

# Print the LaTeX table
cat(paste(latex_table, collapse = "\n"), "\n")

if (exists("saveplt") && saveplt) {
  # Save the LaTeX table to a file
  writeLines(latex_table,
            file.path(base_path_plt, "top_bottom_10_table_iter2.tex"))
} 