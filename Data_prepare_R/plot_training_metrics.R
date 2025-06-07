# Read the CSV files
batch_metrics <- read.csv("~/Documents/Research/HPC/dfs2/mrsec/ML-MD-Peptide/DL_for_Peptide/batch_metrics.csv")
epoch_metrics <- read.csv("~/Documents/Research/HPC/dfs2/mrsec/ML-MD-Peptide/DL_for_Peptide/epoch_metrics.csv")

# Create batch-level plot
p1 <- ggplot(batch_metrics, aes(x = batch, y = loss, color = factor(epoch))) +
    geom_line() +
    scale_color_viridis_d() +
    labs(x = "Batch", y = "Loss", color = "Epoch") +
    theme_minimal() +
    theme(legend.position = "right")

# Create epoch-level plots
p2 <- ggplot(epoch_metrics, aes(x = epoch, y = train_loss)) +
    geom_line() +
    labs(x = "", y = "Train Loss") +
    theme_minimal() +
    theme(axis.title.x = element_blank())

p3 <- ggplot(epoch_metrics, aes(x = epoch, y = learning_rate)) +
    geom_line() +
    labs(x = "", y = "Learning Rate") +
    theme_minimal() +
    theme(axis.title.x = element_blank())

p4 <- ggplot(epoch_metrics, aes(x = epoch, y = max_prob_error)) +
    geom_line() +
    labs(x = "", y = "Max Prob Error") +
    theme_minimal() +
    theme(axis.title.x = element_blank())

p5 <- ggplot(epoch_metrics, aes(x = epoch, y = valid_kl)) +
    geom_line() +
    labs(x = "Epoch", y = "Valid KL") +
    theme_minimal()

# Combine plots vertically
combined_plot <- grid.arrange(p2, p3, p4, p5, ncol = 1)
combined_plot
if (0) {
# Save plots
ggsave("batch_metrics.png", p1, width = 10, height = 6)
ggsave("epoch_metrics.png", combined_plot, width = 10, height = 12)
}