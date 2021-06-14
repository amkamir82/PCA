from PIL import Image

# importing the image / size = 768*1024
filename = "my2.jpg"
image = Image.open(filename)

# image.size returns a 2-tuple
width, height = image.size
print(height)
print(width)

import numpy as np

# transforming the "image" list into ndarray object "npimage"
npimage = np.array(image)

# define an array in a desired form (section 7.5 of the book) / [X1 X2 ... Xn] that Xi represents a [x1, x2, x3] for [R, G, B]
arr = []

# copying the im array into arr in the desired form
for y in range(height - 1):
    for x in range(width - 1):
        arr.append(npimage[y, x])

# transforming the "arr" list into ndarray object "nparr"
nparr = np.array(arr)
from matplotlib import pyplot as plt


def show_data(nparr):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(nparr[:, 0], nparr[:, 1], nparr[:, 2], c='b', marker='.')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    plt.show()


show_data(nparr)


# it receives a 2D ndarray and shows the mean of all 1D arrays in it
def data_mean(nparr):
    result = nparr.sum(axis=0)
    columns = nparr.shape[0]
    return result / columns


def mean_deviation_func(nparr, mean):
    mean_deviation = np.array(nparr - mean)
    return mean_deviation


nparr_mean = data_mean(nparr)
nparr_mean_deviation = mean_deviation_func(nparr, nparr_mean)


def covariance_matrix(nparr_mean_deviation):
    transpose = nparr_mean_deviation.transpose()

    columns = nparr_mean_deviation.shape[0]
    return (1 / (columns - 1)) * np.dot(transpose, nparr_mean_deviation)


covariance_matrix = covariance_matrix(nparr_mean_deviation)

print("variance of Red is:")
print(covariance_matrix[0, 0])
print("variance of Green is:")
print(covariance_matrix[1, 1])
print("variance of Blue is:")
print(covariance_matrix[2, 2])

print("covariance between Red and Green is:")
print(covariance_matrix[0, 1])
print("covariance between Red and Blue is:")
print(covariance_matrix[0, 2])
print("covariance between Blue and Green is:")
print(covariance_matrix[1, 2])


# returns a tuple of (eigenvalues, eigenvectors)
def eigen(covariance_matrix):
    return np.linalg.eig(covariance_matrix)


eigenvalues, eigenvectors = eigen(covariance_matrix)

print("eigenvalues are:")
print(eigenvalues)
print("eigenvectors are:")
print(eigenvectors)


def explained_variance(eigenvalues):
    total = sum(eigenvalues)
    exp_variance = [(i / total) * 100 for i in eigenvalues]
    objects = ('Red', 'Green', 'Blue')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, exp_variance, align='center', alpha=0.5)

    plt.xticks(y_pos, objects)
    plt.ylabel('Percentage')
    plt.title('Explained Variance')


    plt.show()

# explained_variance(eigenvalues)
explained_variance(eigenvalues)

# sorting the eigen pairs in descending order
eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# constructing matrix w
matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
print('Matrix W:\n', matrix_w)


def show_data_2d(nparr):
    # x-axis values
    x = nparr[:, 0]
    # y-axis values

    y = nparr[:, 1]
    # plotting points as a scatter plot
    plt.scatter(x, y, label="pixels", color="green", marker=".", s=30)

    # x-axis label
    plt.xlabel('principle component 1')
    # frequency label
    plt.ylabel('principle component 2')
    # plot title
    plt.title('New Space')
    # showing legend
    plt.legend()

    # function to show the plot
    plt.show()


# mapping the dataset to new space and printing it
new_arr = nparr.dot(matrix_w)
# show_data_2d(new_arr)
show_data_2d(new_arr)

size_image=(height-1)*(width-1)
ok = np.zeros([size_image, 1])
ok1=np.hstack((new_arr, ok))
print(ok1.shape)
new_image = Image.fromarray(np.reshape(ok1, (height-1, width-1, 3)).astype('uint8'))
new_image.save("newok2.jpg")






