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


# Function to read iteration1 data
read_iteration1_data <- function(base_path) {
  data_path <- file.path(base_path, "data/iteration1/training/Sequential_Peptides_edges")
  
  # Read all data files
  iter1_test_results <- read.csv(file.path(data_path, "Test_reg_Transformer_MAE_0.1641775521502461_lr_0.2_bs_1024.csv"))
  iter1_pred_results <- read.csv(file.path(data_path, "ddedge_predict_seqs.csv"))
  iter1_train_data <- read.csv(file.path(data_path, "ddedge_train_seqs.csv"))
  iter1_valid_data <- read.csv(file.path(data_path, "ddedge_valid_seqs.csv"))
  iter1_test_data <- read.csv(file.path(data_path, "ddedge_test_seqs.csv"))
  
  # Prepare error data
  iter1_error_data <- iter1_test_results %>%
    mutate(abs_error = abs(label - predict),
           squared_error = (label - predict)^2) %>%
    select(feature, label, abs_error, squared_error) %>%
    gather(key = "error_type", value = "error", -c(feature, label))
  
  # Prepare distribution data with normalized frequencies
  iter1_dist_data <- bind_rows(
    data.frame(dataset = "Train", value = iter1_train_data$Label),
    data.frame(dataset = "Validation", value = iter1_valid_data$Label),
    data.frame(dataset = "Test", value = iter1_test_data$Label),
    data.frame(dataset = "Predictions", value = iter1_pred_results$Prediction)
  ) %>%
  group_by(dataset) %>%
  mutate(value = value,
         density = n() / sum(n())) %>%
  ungroup()
  
  return(list(
    test_results = iter1_test_results,
    pred_results = iter1_pred_results,
    error_data = iter1_error_data,
    dist_data = iter1_dist_data
  ))
}

# Read data
if (readdata) {
  iter1_data <-  read_iteration1_data("~/Documents/Research/HPC/dfs2/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/")
}



# Plot 1: Scatter plot with points colored by feature
p1 <- ggplot(iter1_data$test_results, aes(x = label, y = predict, color = feature)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  labs(x = "True dEdge", y = "Predicted dEdge", 
       color = "Sequence") +
  plttheme +
  text_theme

# Plot 2: Distribution of dEdge values across different datasets
p2 <- ggplot(iter1_data$dist_data, aes(x = value, color = dataset)) +
  geom_density(alpha = 0.5, fill = NA) +
  labs(x = "dEdge", y = "Normalized Frequency",
       color = "") +
  plttheme +
  text_theme +
  theme(legend.direction = "vertical",
        legend.key.size = unit(8, "pt")) +
  guides(color = guide_legend(nrow = 4, ncol = 1))

# Plot 3: Error vs Label scatter plot with different shapes for error types
p3 <- ggplot(iter1_data$error_data, aes(x = label, y = error, shape = error_type, color = error_type)) +
  geom_point() +
  scale_shape_manual(values = c("abs_error" = 24, "squared_error" = 22),
                     labels = c("Absolute Error", "Squared Error")) +
  scale_color_manual(values = c("abs_error" = "#2ecc71", "squared_error" = "#e74c3c"),
                     labels = c("Absolute Error", "Squared Error")) +
  labs(x = "True dEdge", y = "Error",
       shape = "",
       color = "") +
  ylim(0,1)+
  plttheme +
  text_theme +
  theme(legend.direction  = "vertical",
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

# Display interactive plots
print(ggplotly(p1))
print(ggplotly(p2))
print(ggplotly(p3))

# Save high-resolution plots
if (exists("saveplt") && saveplt) {
  # Save combined plot
  save_plot(file.path(base_path_plt, "iteration1_combined_analysis.png"), 
            combined_plot,
            base_width = 6.5,
            base_height = 2,
            dpi = 1100,
            units = "in")
  
  if (0) {
    # Save individual plots as before
    ggsave(file.path(base_path_plt, "plots/iteration1_ddedge_pred_vs_actual.png"), 
           plot = p1,
           dpi = 1100, 
           width = 32,
           height = 16,
           units = "cm")
    
    ggsave(file.path(base_path_plt, "plots/iteration1_ddedge_error_scatter.png"), 
           plot = p2,
           dpi = 1100, 
           width = 32,
           height = 16,
           units = "cm")
    
    ggsave(file.path(base_path_plt, "plots/iteration1_ddedge_distributions.png"), 
           plot = p3,
           dpi = 1100, 
           width = 32,
           height = 16,
           units = "cm")
  }
}

# Save top and bottom 20 sequences
top_20 <- iter1_data$pred_results %>% 
  arrange(desc(Prediction)) %>% 
  head(20) %>%
  mutate(rank_type = "Top",
         rank = row_number())

bottom_20 <- iter1_data$pred_results %>% 
  arrange(Prediction) %>% 
  head(20) %>%
  mutate(rank_type = "Bottom",
         rank = row_number())

# Combine and format for output
sequences_to_save <- bind_rows(top_20, bottom_20) %>%
  mutate(output_line = sprintf("%s_%d: %s (%.4f)", 
                               rank_type, rank, 
                               Feature, Prediction))

if (0) {
  # Save to file
  writeLines(c("# Top and Bottom 20 Sequences by Predicted dEdge (Iteration 1)",
               "# Format: Rank_Type_Position: Sequence (Predicted_Value)",
               "",
               sequences_to_save$output_line),
             file.path(base_path, "data/iteration1/selected_sequences_tetrapeptide.txt"))
  
  
  # Print analysis results
  print("Top 10 predictions:")
  print(iter1_data$pred_results %>% 
          arrange(desc(Prediction)) %>% 
          head(10))
  
  print("Bottom 10 predictions:")
  print(iter1_data$pred_results %>% 
          arrange(Prediction) %>% 
          head(10))
  
  # Print summary statistics
  print("Summary Statistics for Test Results:")
  print(summary(iter1_data$test_results[c("label", "predict", "MSE", "MAE")]))
  
  print("Overall Model Performance Metrics:")
  print(paste("Average MSE:", mean(iter1_data$test_results$MSE)))
  print(paste("Average MAE:", mean(iter1_data$test_results$MAE)))
  print(paste("R-squared:", unique(iter1_data$test_results$R2)))
  
  # Generate LaTeX table for top and bottom 10
  top_10 <- iter1_data$pred_results %>% 
    arrange(desc(Prediction)) %>% 
    head(10) %>%
    select(Feature, Prediction)
  
  bottom_10 <- iter1_data$pred_results %>% 
    arrange(Prediction) %>% 
    head(10) %>%
    select(Feature, Prediction)
  
  # Create LaTeX table
  latex_table <- c(
    "\\begin{table}[H]",
    "    \\centering",
    "    \\caption{Top 10 and Bottom 10 sequences based on predicted dEdge values.}",
    "    \\label{tab:top_bottom_10_ddedge}",
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
               file.path(base_path_plt, "top_bottom_10_table.tex"))
  } 
  
}