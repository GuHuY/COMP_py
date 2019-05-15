import numpy as np
import matplotlib.pyplot as plt


def Parametric_Classiﬁcation(data, k, x):
    """
    Determine which class has the highest posterior probability for the
    x-value and return the class number.

    Parameters:
        data(np.array): A n × 2 set of (training) input data. Where the ﬁrst
                        column is a one-dimensional set of data, and the
                        second column is the Class, ranging from 0 to k-1.
        k(int): The number of classes.
        x(int): An test input to classify.

    Return:
        (int): The class with the highest posterior probability for x.
    """

    (means, stds) = build_model(data, k)

    posteriors = []
    # Calculate posterior probabilities in ecah class for the input x.
    for class_index in range(k):
        posteriors.append(normfun(x, means[class_index], stds[class_index]))

    # A plot for validation.
    produce_plot(data, means, stds, k)

    return posteriors.index(max(posteriors))


def build_model(data, k):
    """
    Calculate means and variances for each class.

    Parameters:
        data(np.array): A n × 2 set of (training) input data. Where the ﬁrst
                        column is a one-dimensional set of data, and the
                        second column is the Class, ranging from 0 to k-1.
        k(int): The number of classes.

    Returns:
        means(list): The list of mu.
        stds(list): The list of sigma.
    """
    means = []
    stds = []
    for class_index in range(k):
        positions_of_elements = np.where(data[:, 1] == class_index)[0]
        current_class = data[positions_of_elements, 0]
        means.append(np.mean(current_class))
        stds.append(np.std(current_class))
    return means, stds


def normfun(x, mu, sigma):
    """
    Gaussian distribution function.
    """
    return np.exp(-((x-mu)**2)/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))


def produce_plot(data, means, stds, k):
    """
    produce a plot of Gaussian distribution functions.

    Parameters:
        data(np.array): A n × 2 set of (training) input data. Where the ﬁrst
                        column is a one-dimensional set of data, and the
                        second column is the Class, ranging from 0 to k-1.
        means(list): The list of mu.
        stds(list): The list of sigma.
        k(int): The number of classes.
    """
    left = data[:, 0].min()
    right = data[:, 0].max()
    interval = 0.5 * (right-left)
    x = np.linspace(left-interval, right+interval, 1000)
    plt.figure()
    for class_index in range(k):
        y = [normfun(s, means[class_index], stds[class_index]) for s in x]
        plt.plot(x, y, label='k = ' + str(class_index))
    plt.title('Likelihood')
    plt.xlabel('x')
    plt.ylabel('p(x|Ci)')
    plt.legend()
    plt.show()


# iris_1.txt is derivated from iris.txt but replace all ',' with ' ',
# all 'Iris-setosa' with 0, all 'Iris-versicolor' with 1 and all
# 'Iris-virginica' with 2.
iris_1 = np.loadtxt('iris_1.txt')
m_data = iris_1[:, [0, 4]]
print(Parametric_Classiﬁcation(m_data, 3, 5.1))