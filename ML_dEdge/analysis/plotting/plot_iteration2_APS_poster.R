#!/usr/bin/env Rscript

# Custom theme for text sizes
text_theme <- theme(
  plot.title = element_text(size = plot_title_size),
  axis.title = element_text(size = axis_title_size),
  axis.text = element_text(size = axis_text_size),
  legend.title = element_text(size = legend_title_size),
  legend.text = element_text(size = legend_text_size)
)
# Set paths
data_dir <- "~/Documents/Research/HPC/dfs2/mrsec/ML-MD-Peptide/DL_for_Peptide/ML_dEdge/analysis/Data_reshape/"

# Load the binned prediction data
load(file.path(data_dir, "prediction_results.rda"))

# Create the plot
p <- ggplot(dist_data, aes(x = as.numeric(gsub(".*\\(|,.*", "", bin)) + 0.05, 
                           y = normalized_count, 
                           col = length)) +
  geom_line()+
  #geom_density(alpha = 0.5, stat = "identity") +
  #scale_fill_brewer(palette = "Set1") +
  labs(
    x = "dEdge",
    y = "Density",
    fill = "Peptide Length",
    title = ""
  ) +
  plttheme +
  text_theme +
  theme(legend.position = c(0.8,0.8))
print(p)


if (exists("save_plots") && save_plots) {
  # Save the plot
  ggsave(
    filename = file.path(pltsavedir, "iteration2_APS_poster.png"),
    plot = p,
    width  = 3.25,
    height   = 2,
    dpi = 1100
  )
}
