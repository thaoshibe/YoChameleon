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
        if label =='text':
            sns.scatterplot(x=embeddings[indices, 0], y=embeddings[indices, 1], label=label, color=color, alpha=0.01)
        else:
            sns.scatterplot(x=embeddings[indices, 0], y=embeddings[indices, 1], label=label, color=color, alpha=1)

    plt.title(title)
    plt.legend(title="Embedding Type")
    plt.savefig(title)

def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)))
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1])
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], s=100)
    plt.title(title, fontsize=18)
    plt.savefig('umap.png')

if __name__ == '__main__':
    vis_type = 'tsne'  # 'tsne', 'umap'

    # Load embeddings and create labels for each embedding type
    image_embed = torch.load('vocab.pt').data.cpu().float().numpy()[4:8196] #8196
    text_embed = torch.load('vocab.pt').data.cpu().float().numpy()[8196:] #16383
    gen_embed = torch.load('/sensei-fs/users/thaon/ckpt/gen/bo/15-token.pt').data.cpu().float().numpy()[2:]
    recog_embed = torch.load('/sensei-fs/users/thaon/ckpt-before1025/yollava/bo/15-token.pt').data.cpu().float().numpy()[2:]
    together_embed = torch.load('/sensei-fs/users/thaon/ckpt/together/bo/15-token.pt').data.cpu().float().numpy()[2:]
    # Combine the embeddings and labels
    all_embeddings = np.vstack((text_embed, image_embed, gen_embed, recog_embed, together_embed))
    labels = ['text'] * len(text_embed) + ['image'] * len(image_embed) + ['generation-only'] * len(gen_embed) + ['text-only'] * len(recog_embed) + ['together'] * len(together_embed)

    # Apply dimensionality reduction
    start = time.time()
    if vis_type == 'tsne':
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, verbose=True)
        tsne_results = tsne.fit_transform(all_embeddings)
        plot_embeddings(tsne_results, labels, "t-SNE Visualization")
    elif vis_type == 'umap':
        # UMAP\
        # draw_umap(all_embeddings, n_neighbors=100, min_dist=0, n_components=3, metric='cosine', title='UMAP')
        umap_reducer = umap.UMAP(n_components=2, random_state=42, verbose=True)
        umap_results = umap_reducer.fit_transform(all_embeddings)
        plot_embeddings(umap_results, labels, "UMAP Visualization")
    end = time.time()

    print("The elapsed time is : ", end - start)
    print("-"*100)
