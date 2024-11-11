import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import torch
import umap

from sklearn.manifold import TSNE


# Plotting function
def plot_embeddings(embeddings, labels, title):
    # Define unique colors for each embedding type
    colors = {
        'text': 'blue',
        'image': 'green',
        'generation-only': 'red',
        "text-only": 'orange',
        "together": 'purple'
    }
    
    plt.figure(figsize=(10, 8))

    # Plot each embedding type with its color
    for label, color in colors.items():
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        sns.scatterplot(x=embeddings[indices, 0], y=embeddings[indices, 1], label=label, color=color)

    plt.title(title)
    plt.legend(title="Embedding Type")
    plt.savefig(title)


if __name__ == '__main__':
    vis_type = 'umap'  # 'tsne', 'umap'
    embed_len = 100

    # Load embeddings and create labels for each embedding type
    text_emb = torch.load('vocab.pt').data.cpu().float().numpy()[4:8500]
    image_emb = torch.load('vocab.pt').data.cpu().float().numpy()[8500:]
    gen_embed = torch.load('/sensei-fs/users/thaon/ckpt/gen/bo/30-token.pt').data.cpu().float().numpy()
    recog_embed = torch.load('/sensei-fs/users/thaon/ckpt/recog/bo/30-token.pt').data.cpu().float().numpy()
    together_embed = torch.load('/sensei-fs/users/thaon/ckpt/together/bo/30-token.pt').data.cpu().float().numpy()
    # Combine the embeddings and labels
    all_embeddings = np.vstack((text_emb, image_emb, gen_embed, recog_embed, together_embed))
    labels = ['text'] * len(text_emb) + ['image'] * len(image_emb) + ['generation-only'] * len(gen_embed) + ['text-only'] * len(recog_embed) + ['together'] * len(together_embed)

    # Apply dimensionality reduction
    start = time.time()
    if vis_type == 'tsne':
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(all_embeddings)
        plot_embeddings(tsne_results, labels, "t-SNE Visualization")
    elif vis_type == 'umap':
        # UMAP
        umap_reducer = umap.UMAP(n_components=2, random_state=42, verbose=True)
        umap_results = umap_reducer.fit_transform(all_embeddings)
        plot_embeddings(umap_results, labels, "UMAP Visualization")
    end = time.time()

    print("The elapsed time is : ", end - start)
    print("-"*100)
