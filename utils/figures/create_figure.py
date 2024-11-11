import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main(args):
    # Convert inputs to numpy arrays
    x_axis = np.array(args.x_values)
    y_axis = np.array(args.y_values)
    error = np.array(args.error_values)

    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, y_axis, 'o-', color=args.color, linewidth=args.linewidth, markersize=args.markersize)
    plt.fill_between(x_axis, y_axis - error, y_axis + error, color=args.color, alpha=0.2)

    # Set labels and title
    plt.xlabel("Token Length", fontsize=15)
    plt.ylabel("CLIP Image Similarity", fontsize=15)
    plt.ylim(0, 1.1)

    # Aesthetics for an academic-style plot
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    sns.despine()

    # Save the figure as a high-quality image for publication
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    plt.show()

def multiple_plots(args_list):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4)) 
    sns.set_style("whitegrid")
    sns.color_palette()
    for i, args in enumerate(args_list):
        # Convert inputs to numpy arrays
        x_axis = np.array(args["x_values"])
        y_axis = np.array(args["y_values"])
        error = np.array(args["error_values"])
        
        # Plot each subplot
        if args['multiple_y']:
            for index in range(y_axis.shape[0]):
                axs[i].plot(x_axis, y_axis[index], 'o-', linewidth=args["linewidth"], markersize=args["markersize"], label=args["label"][index],
                            color=args["color"][index]
                           )
                axs[i].fill_between(x_axis, y_axis[index] - error, y_axis[index] + error, alpha=0.2,
                                    color=args["color"][index]
                                   )
        else:
            axs[i].plot(x_axis, y_axis, 'o-', linewidth=args["linewidth"], markersize=args["markersize"], label=args["label"],
                        color=args["color"]
                       )
            axs[i].fill_between(x_axis, y_axis - error, y_axis + error, alpha=0.2,
                                color=args["color"]
                               )
        
        # Set labels and title for each subplot
        axs[i].set_xlabel(args['x_label'], fontsize=12)
        axs[i].set_ylabel(args['y_label'], fontsize=12)
        
        # Add legend on the right of each subplot
        axs[i].legend(loc='center left', bbox_to_anchor=(0.01, 1), fontsize=10)
        
        # Grid and despine
        axs[i].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].tick_params(axis='both', labelsize=10)
        sns.despine(ax=axs[i])

    # Overall figure title
    fig.suptitle("Ablation Study: Effectiveness Soft Positive Images", fontsize=17, y=0.9)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjusting layout to fit title
    plt.savefig("multiple_plots_output.pdf", dpi=300)
    print('Saved at multiple_plots_output.png')

# Define the main plotting function
def scatter(args):
    sns.set_style("white")
    # sns.color_palette()
    # Convert inputs to numpy arrays
    x_axis = np.array(args["x_values"])
    y_axis = np.array(args["y_values"])
    error = np.array(args["error_values"])

    # Create the figure
    fig = plt.figure(figsize=(6, 4))

    # Scatter plot for each item in the list
    for i, label in enumerate(args["labels"]):
        plt.scatter(x_axis[i], y_axis[i], label=label, color=args["colors"][i], 
                    marker=args['markers'][i], s=args["markersize"][i])

        # Adding error bars
        # plt.errorbar(x_axis, y_axis, yerr=error, fmt='none', ecolor=args["colors"][i], 
                     # elinewidth=1, capsize=3, alpha=0.7)

    # Set labels and title
    plt.xlabel("Number of Tokens", fontsize=15)
    plt.ylabel("CLIP Image Similarity", fontsize=15)
    plt.xscale('log')  # Set x-axis to log scale
    # plt.ylim(0.55, 0.85)

    # Add a legend
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(0.5, 0.25))

    # Aesthetics for an academic-style plot
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    sns.despine()

    # Save the figure as a high-quality image for publication
    plt.tight_layout()
    plt.savefig(args["output_file"], dpi=300)
    print(f'Saved at {args["output_file"]}')

# args = {
#     "x_values": [16, 1024, 1024, 5120, 64],
#     "y_values": [0.6, 0.76, 0.80, 0.81, 0.82],
#     "error_values": [0.05, 0.05, 0.03, 0.04, 0.6],
#     "labels": ["Chameleon (Text)", "Chameleon (Image)", "GPT-4o (Text)", "GPT-4o (Image)", "YoChamleon (Ours)"],
#     "colors": ['b', 'b', 'r', 'r', 'g'],
#     "markers": ['x', 'X', '+', 'P', '*'],
#     "markersize": [300, 200, 300, 200, 600],
#     "linewidth": 2,
#     "output_file": "scatter_plot.pdf"
# }
# scatter(args)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot academic-style line graph with error bars.")
    
    parser.add_argument("--x_values", nargs='+', type=float, default=[0, 8, 16, 32, 64, 128, 256],
                        help="Values for the x-axis (token lengths).")
    parser.add_argument("--y_values", nargs='+', type=float, default=[0.1, 0.5, 0.3, 0.4, 0.8, 0.9, 0.9],
                        help="Values for the y-axis (accuracies).")
    parser.add_argument("--error_values", nargs='+', type=float, default=[0.05, 0.08, 0.02, 0.03, 0.03, 0.02, 0.03],
                        help="Error values for each y-axis point.")
    parser.add_argument("--output_file", type=str, default="academic_style_plot.pdf",
                        help="Filename for saving the output plot.")
    parser.add_argument("--color", type=str, default="brown", help="Color of the plot line and fill.")
    parser.add_argument("--linewidth", type=float, default=1.5, help="Line width of the plot.")
    parser.add_argument("--markersize", type=float, default=6, help="Marker size of the plot points.")

    args = parser.parse_args()
    main(args)

# Example input for plotting
# args_list = [
#     {
#         "multiple_y": True,
#         "x_values": [0, 5, 10, 15, 20, 25, 30],
#         "y_values": [
#                     [0.4, 0.3, 0.5, 0.55, 0.52, 0.51, 0.50],
#                     [0.4, 0.6, 0.55, 0.57, 0.59, 0.56, 0.56],
#                     [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.82],
#                     ],
#         "x_label": "Training Epochs",
#         "y_label": "CLIP Image Similarity",
#         "error_values": [0.02, 0.005, 0.01, 0.02, 0.02, 0.025, 0.03],
#         "color": ["r", "b", "g"],
#         "linewidth": 2,
#         "markersize": 6,
#         "label": ["Positive Only", "Data Augmentation", "Soft Positive (Ours)"]
#     },
#     {
#         "multiple_y": True,
#         "x_values": [0, 500, 1000, 1500, 2000, 3000],
#         "y_values": [
#                     [0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
#                     [0.6, 0.66, 0.6, 0.63, 0.64, 0.62],
#                     [0.7, 0.75, 0.76, 0.78, 0.8, 0.82],
#                     ],
#         "x_label": "Number of Negative Images",
#         "y_label": "CLIP Image Similarity",
#         "error_values": [0.06, 0.01, 0.02, 0.015, 0.022, 0.01],
#         "color": ["r", "b", "g"],
#         "linewidth": 2,
#         "markersize": 6,
#         "label": ["Positive Only", "Data Augmentation", "Soft Positive (Ours)"]
#     },
#     {
#         "multiple_y": False,
#         "x_values": [0, 8, 16, 32, 64, 128, 256],
#         "y_values": [0.1, 0.5, 0.3, 0.4, 0.8, 0.9, 0.9],
#         "x_label": "Number of Learnable Token",
#         "y_label": "CLIP Image Similarity",
#         "error_values": [0.05, 0.08, 0.02, 0.03, 0.03, 0.02, 0.03],
#         "color": "g",
#         "linewidth": 2,
#         "markersize": 6,
#         "label": "Soft Positive (Ours)"
#     },
# ]

# # Call the function to create the plot
# multiple_plots(args_list)
