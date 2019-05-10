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

# Set some parameters for the console
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)

# TEST DATA
'''X = pd.DataFrame(
    data=[
        [1, 6, 7, 2, 5, 7, 6, 3, 6, 7],
        [5, 3, 2, 4, 4, 4, 2, 4, 4, 3],
        [6, 1, 1, 5, 2, 1, 1, 7, 1, 1],
        [7, 1, 2, 7, 2, 1, 2, 2, 2, 2],
        [2, 5, 4, 3, 5, 6, 5, 2, 6, 6],
        [3, 4, 4, 3, 5, 4, 5, 1, 7, 5]
    ],

    columns=['E1 fruity', 'E1 woody', 'E1 coffee',
             'E2 red fruit', 'E2 roasted', 'E2 vanillin', 'E2 woody',
             'E3 fruity', 'E3 butter', 'E3 woody'],

    index=['Wine {}'.format(i + 1) for i in range(6)]
)
X['Oak type'] = [1, 2, 2, 2, 1, 1]

X1 = X.loc[:, 'E1 fruity':'E1 coffee']
X2 = X.loc[:, 'E2 red fruit':'E2 woody']
X3 = X.loc[:, 'E3 fruity':'E3 woody']

attribute_matrices = {
    'Expert1': center_and_normalize(X1),
    'Expert2': center_and_normalize(X2),
    'Expert3': center_and_normalize(X3)
}'''


####################
# Data preparation #
####################

def center_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Center each column in a data frame (column minus column mean).

    :param df: data frame of which columns are to be centered and normalized
    :return: resulting data frame
    """

    # These are the attributes, if all went well
    columns = df.columns

    # Center all attribute columns
    for col in columns:
        # Make sure they're floats
        df[col] = df[col].astype(float)

        # Center by subtracting column mean from each column
        df[col] = df[col] - df[col].mean()

    # Normalize columns so sum of squared elements is 1
    for col in columns:
        # Calculate sum of squares
        sum_of_squares = np.sum(df[col] ** 2)

        # Divide by square root of sum of squares
        df[col] = df[col] / np.sqrt(sum_of_squares)

    # Return that shit whassup
    return df


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

    :param attribute_matrix (pd.DataFrame): data frame containing attribute scores (columns)
            for different observations (rows)
    :return
        - Normalized data frame (pd.DataFrame)
        - Principal Component Analysis object (PCA)
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

    :param attribute_matrix (pd.DataFrame): data frame containing attribute scores,
            resulting from previous pca.
    :param scree_plot (bool): determines whether to show a Scree plot
    :return
        - Global factor score matrix: each row represents an observation and each column a component.
        - Eigenvalues
        - U, Delta, Vh: singular value decomposition of global matrix.
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
    :return individual factor score matrix
    """

    # Mass matrix
    n_obs = len(attribute_matrix)
    M = 1 / n_obs * np.identity(n=n_obs)
    Mnew = inv(sqrtm(M))

    # Projection matrix
    P = matmul(matmul(Mnew, U), inv(diag(Delta)))

    # Scale single expert with all experts of global solution by multiplying by T
    T = len(attribute_matrices)

    # Attribute matrix
    Z = attribute_matrix.values

    # Projection of scores
    ZZt = matmul(Z, transpose(Z))
    F = T * matmul(ZZt, P)

    return F


def get_partial_inertias(Vh: np.ndarray, eigenvals: np.ndarray, attribute_matrices: dict):
    labels = []

    V = Vh.transpose()

    J = [len(attribute_matrices[pp].columns) for pp in attribute_matrices]

    n_obs = len(attribute_matrices[list(attribute_matrices.keys())[0]])
    n_groups = len(J)

    partial_inertias = np.ndarray(shape=(n_groups, n_obs))

    for i in range(n_obs):
        start = 0
        stop = 0
        lam = eigenvals[i]
        for j in range(n_groups):
            stop += J[j]
            q = np.sum(V[start:stop, i] ** 2)
            partial_inertias[j, i] = lam * q
            start = stop

    return partial_inertias


###
#
###


#################
# Visualization #
#################

def plot_projection_global(global_factor_scores: np.ndarray, components=(1, 2)):
    """
    Plot the different observations (in this case, different content) on the axes
    of the principal components, as determined by the final step PCA.

    :param global_factor_scores: matrix with global factor scores, resulting from final step PCA.
    :param components: which component loadings to plot.
    :return void
    """

    # Gather coordinates
    x = global_factor_scores[:, components[0] - 1]
    y = global_factor_scores[:, components[1] - 1]

    # Set limits for axes
    xmax, ymax = np.max(np.abs(x)) * 1.1, np.max(np.abs(y)) * 1.1
    xmin, ymin = -xmax, -ymax
    scale_factor = np.max([xmax, ymax]) / 50
    print(scale_factor)

    # Plot with seaborn
    colors = sns.color_palette("pastel", n_colors=6)
    sns.set(style="white", rc={'figure.figsize': (5, 5)})
    # ax = sns.scatterplot(x, y, hue=content, legend=None)
    ax = sns.scatterplot(x, y, color='white')
    sns.despine(ax=ax, left=True, bottom=True)

    ax.set(xlabel='Principal component ' + str(components[0]),
           ylabel='Principal component ' + str(components[1]),
           xlim=(xmin, xmax), ylim=(ymin, ymax))

    # Add the labels
    dodge = scale_factor

    for idx, coord in enumerate(zip(x, y)):
        ax.text(x=coord[0] + dodge, y=coord[1] + dodge, s=content[idx].capitalize(), fontsize=12)
        plt.arrow(x=0, y=0, dx=coord[0], dy=coord[1], color=colors[idx], head_width=scale_factor)

    plt.show()

    return ax


def plot_projection_individual(individual_factor_scores: dict, global_factor_scores: np.ndarray, components=(1, 2),
                               which_pp=None):
    """
    Plot the different observations (in this case, different content) on the axes
    of the principal components, as determined by the final step PCA. In addition,
    visualize how individual ratings stack up with the global structure.

    :param individual_factor_scores: matrices with individual factor scores (normalized by first eigenvalue).
    :param global_factor_scores: matrix with global factor scores, resulting from final step PCA.
    :param components: which component loadings to plot.
    :param which_pp: which participants (observers; groups of variables) to include in the plot.
    :return: void
    """

    # Gather coordinates of indicated components
    x = global_factor_scores[:, components[0] - 1]
    y = global_factor_scores[:, components[1] - 1]

    # Plot with seaborn
    sns.set(style="white", rc={'figure.figsize': (5, 5)})

    ax = sns.scatterplot(x, y, color='black', marker='o', s=100)

    ax.set(xlabel='Principal component ' + str(components[0]),
           ylabel='Principal component ' + str(components[1]))

    plt.title("Partial analyses", fontweight='bold')

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

        # Add points
        scores = individual_factor_scores[pp]
        x_ind = scores[:, components[0] - 1]
        y_ind = scores[:, components[1] - 1]
        sns.scatterplot(x_ind, y_ind, color=colors[idx])

        # Add lines
        for x0, y0, x1, y1 in zip(x, y, x_ind, y_ind):
            plt.plot([x0, x1], [y0, y1], color=colors[idx])

    sns.despine(trim=True, offset=10)

    plt.show()


def plot_variable_loadings(attribute_matrices: dict, projections: dict, global_factor_scores: np.ndarray,
                           components=(1, 2), which_pp=None):
    """
    Plot loadings of original variables on principal components, as determined by the correlation.

    :param individual_factor_scores: matrices with individual factor scores (normalized by first eigenvalue).
    :param global_factor_scores: matrix with global factor scores, resulting from final step PCA.
    :param components: which component loadings to plot.
    :param which_pp: which participants (observers; groups of variables) to include in the plot.
    :return: void
    """

    # If left unspecified, plot all observer data
    if which_pp is None:
        which_pp = attribute_matrices.keys()

    # Color palette
    colors = sns.color_palette("pastel", n_colors=len(which_pp))

    # Gather PC coordinates for global
    x = global_factor_scores[:, components[0] - 1]
    y = global_factor_scores[:, components[1] - 1]

    # Calculate correlations, loop over observers
    for idx, pp in enumerate(which_pp):

        x_coordinates, y_coordinates = [], []
        variable_names = list(attribute_matrices[pp].columns)

        # Gather PC coordinates for individual
        first_pc = projections[pp][:, components[0] - 1]
        second_pc = projections[pp][:, components[1] - 1]

        # Correlation between global and individual PCs
        PC00 = np.corrcoef(x=x, y=first_pc)[0, 1]
        PC01 = np.corrcoef(x=y, y=first_pc)[0, 1]
        PC10 = np.corrcoef(x=x, y=second_pc)[0, 1]
        PC11 = np.corrcoef(x=y, y=second_pc)[0, 1]

        # Loop over variables
        for var in attribute_matrices[pp]:
            # Original variables
            x_ind = attribute_matrices[pp][var].values
            y_ind = attribute_matrices[pp][var].values

            # Correlation between PCs an old variables
            rx = np.corrcoef(x=x, y=x_ind)[0, 1]
            ry = np.corrcoef(x=y, y=y_ind)[0, 1]

            # Only print correlations > sqrt(.5)
            plot_yn = True if (rx ** 2 + ry ** 2) ** (1 / 2) > .5 ** (1 / 2) else False

            if plot_yn:
                x_coordinates.append(rx)
                y_coordinates.append(ry)

        # Plot with seaborn
        dodge = .03
        sns.set(style="white", rc={'figure.figsize': (5, 5)})

        # Prepare canvas
        plt.arrow(x=-1, y=0, dx=2, dy=0, color='grey', head_width=.05, zorder=-1, length_includes_head=True)
        plt.arrow(x=0, y=-1, dx=0, dy=2, color='grey', head_width=.05, zorder=-1, length_includes_head=True)
        circle = plt.Circle((0, 0), 1, color='lightgrey', fill=False)

        ax = sns.scatterplot(x_coordinates, y_coordinates, color=colors[idx])

        plt.plot(PC00, PC01, 'o', markerfacecolor='none', markeredgecolor=colors[idx], color=colors[idx], ms=10,
                 zorder=2)
        plt.plot(PC10, PC11, 'o', markerfacecolor='none', markeredgecolor=colors[idx], color=colors[idx], ms=10,
                 zorder=2)
        ax.text(x=PC00 + 2 * dodge, y=PC01 + 2 * dodge, s="PC" + str(components[0]), fontsize=10)
        ax.text(x=PC10 + 2 * dodge, y=PC11 + 2 * dodge, s="PC" + str(components[1]), fontsize=10)

        for i in range(len(x_coordinates)):
            ax.text(x=x_coordinates[i] + dodge, y=y_coordinates[i] + dodge, s=variable_names[i], fontsize=10,
                    color='darkgrey')
        ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        ax.set(xlabel='Principal component ' + str(components[0]),
               ylabel='Principal component ' + str(components[1]))
        ax.add_artist(circle)

        plt.title("Correlation plot - " + pp, fontweight='bold')

        sns.despine(left=True, bottom=True)

        plt.show()

    return ax


def plot_partial_inertias(partial_inertias: np.ndarray, components=(1, 2)):
    """
    Plot partial intertia of observers.

    :param partial_inertias: matrix with partial intertias of each observer. matrices with individual factor scores (normalized by first eigenvalue).
    :param components: which component loadings to plot.
    :return: void
    """

    sns.set_style('white')

    x = partial_inertias[:, components[0] - 1]
    y = partial_inertias[:, components[1] - 1]

    ax = sns.scatterplot(x=x, y=y)

    ax.set(xlabel='Principal component ' + str(components[0]),
           ylabel='Principal component ' + str(components[1]))
    plt.title("Partial intertia plot - Components " + str(components), fontweight='bold')

    dodge = .005

    for pp in range(x.shape[0]):
        ax.text(x=x[pp] + dodge, y=y[pp] + dodge, s=str(pp + 1), fontsize=10)

    sns.despine()

    plt.show()


if __name__ in ['__main__', 'builtins']:

    # Load the processed data files
    df_general = pd.read_csv("../data/df_general.csv", sep=";", decimal=",")
    attribute_matrices = dict(np.load("../data/attributes.npy").item())

    # Set indices
    content = ['plank', 'crow', 'canyon', 'jurassic', 'spacewalk', 'limit']
    df_general = df_general.set_index(['pp'], inplace=False)[content]

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
        projections[pp] = project_onto_global(attribute_matrix=normalized_attribute_matrices[pp],
                                              global_factor_scores=F, U=U,
                                              Delta=Delta, Vh=Vh)

    # Plot content
    plot_projection_global(global_factor_scores=F, components=(1, 2))
    plot_projection_global(global_factor_scores=F, components=(2, 3))

    # Plot observer stack-up
    plot_projection_individual(individual_factor_scores=projections, global_factor_scores=F, components=(1, 2),
                               which_pp=None)

    # Plot correlations with original variables
    ax = plot_variable_loadings(attribute_matrices=normalized_attribute_matrices, projections=projections,
                                global_factor_scores=F, which_pp=None)

    # Plot partial inertias - how do observers stack up against the total?
    partial_inertias = get_partial_inertias(Vh=Vh, eigenvals=eigenvals, attribute_matrices=attribute_matrices)
    plot_partial_inertias(partial_inertias=partial_inertias)

    # TEMP
    hedonic_scores = df_general.transpose()
    for pp in projections:
        temp = pd.DataFrame(data=projections[pp], index=content,
                            columns=['PC' + str(i + 1) for i in range(6)]).reset_index()
        temp['pp'] = int(pp)
        temp['hedonic'] = hedonic_scores.reset_index()[int(pp)]
        print(temp)
        projections[pp] = temp
    data_long = pd.concat(projections.values(), axis=0)
    data_long.to_csv("for_R.csv", sep=',', decimal='.')

    # PrefMFA: combining external matrix (E) and hedonic matrix (H)
    E, _ = separate_pca(
        center_and_normalize(pd.DataFrame(data=F, index=content, columns=['PC' + str(i + 1) for i in range(6)])))
    H, _ = separate_pca(attribute_matrix=center_and_normalize(df_general.transpose()))
    E_and_H = pd.concat([E, H], axis=1)
    F2, eigenvals2, U2, Delta2, Vh2 = global_pca(attribute_matrix=E_and_H, scree_plot=True)

    result = pd.DataFrame(data=F2, index=content, columns=['PC' + str(i + 1) for i in range(6)])

    # Plot content
    plot_projection_global(global_factor_scores=result.values, components=(1, 2))

    plot_variable_loadings(attribute_matrices={'Hedonic': H},
                           projections={'Hedonic': project_onto_global(H, F2, U2, Delta2, Vh2)},
                           global_factor_scores=F2,
                           components=(2, 3))
