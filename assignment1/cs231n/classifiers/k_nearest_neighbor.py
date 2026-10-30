from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                diff = X[i] - self.X_train[j]          
                dists[i, j] = np.sqrt(np.sum(diff ** 2))
                
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        train_norm_seq = np.sum(self.X_train ** 2, axis=1)
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # 我们需要在这个循环里得到单个测试样本对所有训练样本的L2距离向量，形状为[1, num_train]
            # 对于单个测试样本和单个训练样本的L2距离，可以表达为||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x * y
            # 因此我们可以推广到单个测试样本对所有训练样本的L2距离向量计算，向量的每一个位置上的元素为[||x||^2 + ||y_j||^2 - 2 * y * x^T] y 为 [num_train, D], x为[1, D]，这里numpy的规则是，只要@左边的维度和右边的维度相同即可计算消去
           
            x = X[i]
            dists[i] = np.sum(x**2) + train_norm_seq - 2 * (self.X_train @ x)

        dists = np.sqrt(dists)
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        test_norm_seq = np.sum(X**2, axis=1) # the shape is [test_num,]
        train_norm_seq = np.sum(self.X_train**2, axis=1) # the shape is [train_num,]

        # and finally we'll get dists in shape [test_num, train_num]
        a = test_norm_seq[:, np.newaxis]
        b = train_norm_seq[np.newaxis, :]
        norm_sum = a + b # using broadcast, we get sum in shape [test_num, train_num]

        # X shape [test_num, D], X_train shape [train_num, D]
        inner_product = X @ self.X_train.T

        dists = norm_sum - 2 * inner_product
        dists = np.sqrt(dists)
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            min_to_max_distance_train_idx = np.argsort(dists[i])
            nearest_k_train_idx = min_to_max_distance_train_idx[:k]
            closest_y = self.y_train[nearest_k_train_idx]

            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            labels_count_map = {}
            first_pos = {}
            for pos, idx in enumerate(nearest_k_train_idx):
                label = self.y_train[idx]
                labels_count_map[label] = labels_count_map.get(label, 0) + 1
                if label not in first_pos:
                    first_pos[label] = pos
            
            best_label = min(labels_count_map, key=lambda l: (-labels_count_map[l], first_pos[l]))
            y_pred[i] = best_label
        return y_pred
