import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_Cij(sample_label, neighbours_label_values):
    num_common_labels = np.count_nonzero(neighbours_label_values == sample_label)
    k = len(neighbours_label_values)
    return (k - num_common_labels)/k


# for each label lj we compute the proportion of neighbours having opposite class with respect to the class of the
# instance and store the result in the matrix C
def get_C(y, indices):
    C = np.zeros(y.shape)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            C[i, j] = get_Cij(y[i, j], y[indices[i, 1:], j])
    return C


# aggregate the values in C per training example in order to arrive at a singe sampling weight wi
def get_weight_per_example(y, C):
    w = np.zeros(y.shape[0])
    sum_of_non_out_minority_examples_per_example = np.zeros(y.shape[1])
    for j in range(y.shape[1]):
        for i in range(y.shape[0]):
            if y[i, j] == 1 and C[i, j] < 1:
                sum_of_non_out_minority_examples_per_example[j] += C[i, j]

    for i in range(y.shape[0]):
        sum = 0
        for j in range(y.shape[1]):
            if y[i, j] == 1 and C[i, j] < 1:
                sum += C[i, j] / sum_of_non_out_minority_examples_per_example[j]
        w[i] = sum

    return w


def mlsol_main(X, y, perc_gen_instances = 0.3, k = 5):

    # number of instances to generate
    gen_num = X.shape[0] * perc_gen_instances

    X_aug, y_aug = X.copy, y.copy

    # find the kNN of each instance
    # the sklearn implementation suggests that for small number of neighbours the brute force method is comparable
    # to the more computationally efficient methods (kd_tree, ball_tree)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(X)
    distances, indices = nbrs.kneighbors(X)

    C = get_C(y, indices)

    w = get_weight_per_example(y, C)




