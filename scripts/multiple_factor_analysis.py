# -*- coding: utf-8 -*-

"""
   __  __________
  /  |/  / __/ _ |
 / /|_/ / _// __ |
/_/  /_/_/ /_/ |_|

Multiple Factor Analysis
in support of OPQ methodology

-- Coded by Wouter Durnez
-- mailto:Wouter.Durnez@UGent.be
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import matmul, transpose, diag, identity
from numpy.linalg import svd, inv
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA


#################
# Decomposition #
#################

def separate_pca(attribute_matrix: pd.DataFrame) -> (pd.DataFrame, PCA):
    """
    First layer of PCA in Multiple Factor Analysis method.
    ---
    Takes a data frame, consistent of attribute columns (that should
    already by centered and normalized!), calculate pca, and normalize
    data frame using largest singular value.

    Returns:
        - Normalized data frame
        - PCA
    """

    # Apply pca to attribute matrix
    pca = PCA()
    pca.fit(attribute_matrix.values)

    # Normalize attribute matrix based on first (largest) singular value
    normalized_matrix = attribute_matrix / pca.singular_values_[0]

    # Return new df and pca
    return normalized_matrix, pca


def global_pca(attribute_matrix: pd.DataFrame, scree_plot=False) -> (pd.DataFrame, np.ndarray):
    """
    Second layer of PCA in Multiple Factor Analysis method.
    ---
    Takes a data frame, and applies singular value decomposition to
    calculate global factor scores

    :param: attribute_matrix (pd.DataFrame): data frame containing attribute scores,
            resulting from previous pca.
    :param: scree_plot (bool): determines whether to show a Scree plot

    :return:
        - Global factor score matrix
        - Eigenvalues
    """

    # Singular value decomposition
    U, Delta, Vh = svd(a=attribute_matrix)

    # Number of observations and columns
    n_obs = len(attribute_matrix)

    # Eigenvalues (square of singular values)
    eigenvals = Delta ** 2

    # Scree plot
    if scree_plot:
        x = np.arange(n_obs) + 1
        colors = sns.color_palette("pastel", n_colors=6)
        ax = sns.lineplot(x, eigenvals, color=colors[0])
        ax = sns.scatterplot(x, eigenvals, color=colors[1], s=100)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        plt.show()

    # Mass matrix
    M = 1 / n_obs * identity(n=n_obs)

    # M to the power -1/2
    Mnew = inv(sqrtm(M))
    print(Mnew)

    # Global factor scores
    F = matmul(matmul(Mnew, U), diag(Delta))

    return F, eigenvals, U, Delta, Vh


def project_onto_global(attribute_matrix: pd.DataFrame, global_factor_scores: np.ndarray, U: np.ndarray,
                        Delta: np.ndarray, Vh: np.ndarray) -> np.ndarray:
    """
    See how individual observers stack up with global structure, by projecting their
    attribute matrix onto the global factor matrix.

    :param attribute_matrix: individual observer data
    :param global_factor_scores: global factor score matrix
    :param U, Delta, Vh: results from singular value decomposition
    :return: individual factor score matrix
    """

    # Mass matrix
    n_obs = len(attribute_matrix)
    M = 1 / n_obs * np.identity(n=n_obs)
    Mnew = inv(sqrtm(M))

    # Projection matrix
    P = matmul(matmul(Mnew, U), inv(diag(Delta)))

    # Scale single expert with all experts of global solution by multiplying by T
    T = len(attribute_matrices)
    T = 6

    # Attribute matrix
    Z = attribute_matrix.values

    # Projection of scores
    ZZt = matmul(Z, transpose(Z))
    F = T * matmul(ZZt, P)

    return F


#################
# Visualization #
#################

def plot_projection_global(global_factor_scores: np.ndarray, components=(1, 2)):
    """
    Plot the different observations (in this case, different content) on the axes
    of the principal components, as determined by the final step PCA.

    :param global_factor_scores: matrix with global factor scores, resulting from final step PCA.
    :param components: which component loadings to plot.
    :return: void
    """

    # Gather coordinates
    x = global_factor_scores[:, components[0] - 1]
    y = global_factor_scores[:, components[1] - 1]

    # Plot with seaborn
    colors = sns.color_palette("pastel", n_colors=6)
    sns.set(style="white", rc={'figure.figsize': (5, 5)})
    ax = sns.scatterplot(x, y, color='white')
    sns.despine(left=True, bottom=True, trim=True, offset=10)

    ax.set(xlabel='Principal component ' + str(components[0]),
           ylabel='Principal component ' + str(components[1]))

    # Add the labels
    dodge = .3
    content = ["plank", "crow", "canyon", "jurassic", "spacewalk", "limit"]

    for idx, coord in enumerate(zip(x, y)):
        ax.text(x=coord[0] + dodge, y=coord[1] + dodge, s=content[idx].capitalize(), fontsize=12)

        plt.arrow(x=0, y=0, dx=coord[0], dy=coord[1], color=colors[idx], width=.08)

    plt.show()


def plot_projection_individual(individual_factor_scores: dict, global_factor_scores: np.ndarray, components=(1, 2),
                               which_pp=None):
    """
    Plot the different observations (in this case, different content) on the axes
    of the principal components, as determined by the final step PCA. In addition,
    visualize how individual ratings stack up with the global structure.

    :param individual_factor_scores:
    :param global_factor_scores: matrix with global factor scores, resulting from final step PCA.
    :param components: which component loadings to plot.
    :return: void
    """
    # Gather coordinates
    x = global_factor_scores[:, components[0] - 1]
    y = global_factor_scores[:, components[1] - 1]

    # Plot with seaborn
    sns.set(style="white", rc={'figure.figsize': (5, 5)})
    ax = sns.scatterplot(x, y, color='black')
    sns.despine(left=True, bottom=True, trim=True, offset=10)

    ax.set(xlabel='Principal component ' + str(components[0]),
           ylabel='Principal component ' + str(components[1]))

    # Add the labels
    dodge = .3
    content = ["plank", "crow", "canyon", "jurassic", "spacewalk", "limit"]

    for idx, coord in enumerate(zip(x, y)):
        ax.text(x=coord[0] + dodge, y=coord[1] + dodge, s=content[idx].capitalize(), fontsize=12)

    # If left unspecified, plot all observers
    if which_pp is None:
        which_pp = individual_factor_scores.keys()

    # Loop over observers and add to plot
    colors = sns.color_palette("pastel", n_colors=len(which_pp))

    for idx, pp in enumerate(which_pp):
        scores = individual_factor_scores[pp]
        x_ind = scores[:, components[0] - 1]
        y_ind = scores[:, components[1] - 1]

        ax = sns.scatterplot(x_ind, y_ind, color=colors[idx])

    plt.show()


'''def item_map(cluster1: object,
             cluster2: object,
             pca: PCA,
             style="whitegrid"):
    """Plot attributes on principal component axes."""

    # Get the points we need
    points = pd.DataFrame(pca.components_[[cluster1, cluster2]])
    points.columns = df_attributes.columns
    points = points.transpose()
    points.reset_index(inplace=True)

    # Plot the points
    sns.set(style=style, rc={'figure.figsize': (12, 12)})
    ax = sns.scatterplot(x=0, y=1, data=points)
    ax.set(xlabel='Principal component ' + str(cluster1 + 1),
           ylabel='Principal component ' + str(cluster2 + 1))

    # Add the labels
    for i, p in points.iterrows():
        ax.text(x=p[0], y=p[1], s=p['name'], fontsize=6)

    plt.show()'''

if __name__ in ['__main__', 'builtins']:

    # Load the processed data files
    df_general = pd.read_csv("../data/df_general.csv", sep=";", decimal=",")
    attribute_matrices = dict(np.load("../data/attributes.npy").item())

    # Set indices
    df_general.set_index(['pp'], inplace=True)

    # Normalize all attribute matrices on first singular value
    normalized_attribute_matrices = {}

    # Perform first step of MFA
    for pp in attribute_matrices:
        # Calculate normalized matrices (possible to return PCA object)
        normalized_attribute_matrices[pp], \
        _ = separate_pca(attribute_matrix=attribute_matrices[pp])

    # Combine in big matrix, and make sure they're all floats
    df_attributes = pd.concat(list(normalized_attribute_matrices.values()), axis=1).astype(float)

    # Perform second step of MFA
    F, eigenvals, U, Delta, Vh = global_pca(attribute_matrix=df_attributes, scree_plot=True)

    # Ratio explained variance
    ratio_explained_variance = eigenvals / np.sum(eigenvals)

    # Partial analyses
    projections = {}

    for pp in attribute_matrices:
        # Project each attribute matrix onto global structure
        projections[pp] = project_onto_global(attribute_matrix=attribute_matrices[pp], global_factor_scores=F, U=U,
                                              Delta=Delta, Vh=Vh)

    # Plot content
    plot_projection_global(global_factor_scores=F, components=(1, 2))
    plot_projection_global(global_factor_scores=F, components=(2, 3))

    # Plot observer stack-up
    plot_projection_individual(individual_factor_scores=projections, global_factor_scores=F, components=(1, 2),
                               which_pp=None)

    '''pca = PCA()
    pca.fit(X=df_attributes)

    # Check out first two
    sns.set(style="white")
    item_map(cluster1=0, cluster2=1, pca=pca)'''
