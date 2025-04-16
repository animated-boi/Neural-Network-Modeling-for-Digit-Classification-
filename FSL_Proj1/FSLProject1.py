import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


train5 = scipy.io.loadmat('training_data_5.mat')
train6 = scipy.io.loadmat('training_data_6.mat')
test5 = scipy.io.loadmat('testing_data_5.mat')
test6 = scipy.io.loadmat('testing_data_6.mat')

vect_img5 = [image_array.reshape(-1) for image_array in train5['train_data_5']]
vect_img6 = [image_array.reshape(-1) for image_array in train6['train_data_6']]

test_vect5 = [image_array.reshape(-1) for image_array in test5['test_data_5']]
test_vect6 = [image_array.reshape(-1) for image_array in test6['test_data_6']]


def image_normalization (images_vectorized):
    e=1e-10
    mean_values = np.mean(images_vectorized, axis=0)
    std_values = np.std( images_vectorized, axis=0)
    images_normalized = (images_vectorized - mean_values) / (std_values+e)
    return images_normalized, mean_values, std_values

normal5, norm5m, norm5std = image_normalization(vect_img5)
normal6, norm6m, norm6std = image_normalization(vect_img6)

normtest5 = (test_vect5 - norm5m) / (norm5std + 1e-10)
normtest6 = (test_vect6 - norm6m) / (norm6std + 1e-10)
print("Normalising Finished!")
print("Normalized set for 5 :")
print(normtest5 )
print("\n \n \n")
print("Normalized set for 6 :")
print(normtest6)
print("\n \n \n \n \n \n")

# Assuming you have two different data sets for two classes: class_0_data and class_1_data
# Create labels or identifiers for each class
class_0_labels = np.zeros(len(normal5))  # Use 0 to label class 0
class_1_labels = np.ones(len(normal6))   # Use 1 to label class 1

# Combine the data and labels for each class
combined_data = np.vstack((normal5, normal6))
combined_labels = np.concatenate((class_0_labels, class_1_labels))

# Create labels or identifiers for each class
class_0_labelsTest = np.zeros(len(normtest5))  # Use 0 to label class 0
class_1_labelsTest = np.ones(len(normtest6))   # Use 1 to label class 1

# Combine the data and labels for each class
combined_dataTest = np.vstack((normtest5, normtest6))
combined_labelsTest = np.concatenate((class_0_labelsTest, class_1_labelsTest))

# Compute the covariance matrix for the combined data
covariance_matrix = np.cov(combined_data, rowvar=False)

# Perform eigenanalysis (use a library or built-in function)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Sort eigenvectors in descending order of eigenvalues to identify principal components
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
print("The covarience Matrix is :")
print(covariance_matrix)
print("\n \n \n")


# Assuming you have already performed PCA, obtained principal components, and separated data into class-specific arrays if needed

# Project the training and testing data onto the first two principal components
train_proj_2d = np.dot(combined_data, eigenvectors[:, :2])
test_proj_2d = np.dot(combined_dataTest, eigenvectors[:, :2])  # Use your testing data here

# Create scatter plots to visualize the 2-D space
plt.figure(figsize=(8, 6))

plt.scatter(train_proj_2d[combined_labels == 0][:, 0], train_proj_2d[combined_labels == 0][:, 1], label='Training Class 0', c='b', marker='o')

# Plot training data for class 1 (use red color)
plt.scatter(train_proj_2d[combined_labels == 1][:, 0], train_proj_2d[combined_labels == 1][:, 1], label='Training Class 1', c='r', marker='x')

# # Plot testing data for class 0 (use green color)
#plt.scatter(test_proj_2d[combined_labelsTest == 0][:, 0], test_proj_2d[combined_labelsTest == 0][:, 1], label='Testing Class 0', c='g', marker='s')

# # Plot testing data for class 1 (use yellow color)
#plt.scatter(test_proj_2d[combined_labelsTest == 1][:, 0], test_proj_2d[combined_labelsTest == 1][:, 1], label='Testing Class 1', c='y', marker='^')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2-D Projection of Training and Testing Data')
plt.legend()
plt.show()




# Estimate the parameters for class 0 (Training)
mean_class_0 = np.mean(train_proj_2d[combined_labels == 0][:, 0], axis=0)
covariance_matrix_class_0 = np.cov(train_proj_2d[combined_labels == 0][:, 0], rowvar=False)

# Estimate the parameters for class 1 (Testing)
mean_class_1 = np.mean(train_proj_2d[combined_labels == 1][:, 1], axis=0)
covariance_matrix_class_1 = np.cov(train_proj_2d[combined_labels == 1][:, 1], rowvar=False)

# Create multivariate normal distributions for each class
class_0_distribution = multivariate_normal(mean=mean_class_0, cov=covariance_matrix_class_0)
class_1_distribution = multivariate_normal(mean=mean_class_1, cov=covariance_matrix_class_1)



#Assuming you have already performed PCA, obtained principal components, and separated data into class-specific arrays if needed

# Project the training data onto the first two principal components
train_proj_2d = np.dot(combined_data, eigenvectors[:, :2])

# Separate training data into class 0 and class 1
train_class_0 = train_proj_2d[combined_labels == 0]
train_class_1 = train_proj_2d[combined_labels == 1]

# Estimate the parameters for class 0 (Training)
mean_class_0 = np.mean(train_class_0, axis=0)
covariance_matrix_class_0 = np.cov(train_class_0, rowvar=False)

# Estimate the parameters for class 1 (Training)
mean_class_1 = np.mean(train_class_1, axis=0)
covariance_matrix_class_1 = np.cov(train_class_1, rowvar=False)

# Create multivariate normal distributions for each class
class_0_distribution = multivariate_normal(mean=mean_class_0, cov=covariance_matrix_class_0)
class_1_distribution = multivariate_normal(mean=mean_class_1, cov=covariance_matrix_class_1)





# Assuming you have already estimated the parameters and created the multivariate normal distributions
# Also, assuming you have the training and testing data (train_proj_2d and test_proj_2d)

# Classify data points based on maximum posterior probability
def classify_data(data, distribution_0, distribution_1):
    posterior_prob_class_0 = distribution_0.pdf(data)
    posterior_prob_class_1 = distribution_1.pdf(data)
    return (posterior_prob_class_1 > posterior_prob_class_0).astype(int)

# Classify training data
train_predictions = classify_data(train_proj_2d, class_0_distribution, class_1_distribution)
# Classify testing data
test_predictions = classify_data(test_proj_2d, class_0_distribution, class_1_distribution)

# Calculate accuracy
def calculate_accuracy(predictions, true_labels):
    correct = (predictions == true_labels).sum()
    total = len(true_labels)
    accuracy = correct / total
    return accuracy

# Assuming you have true labels for training and testing data
train_labels = combined_labels  # Assuming you've kept the original training labels
test_labels = combined_labelsTest  # Assuming you've renamed test_labels

train_accuracy = calculate_accuracy(train_predictions, train_labels)
test_accuracy = calculate_accuracy(test_predictions, test_labels)
print("\n \n \n")
print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
print("\n")
print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))
print("\n \n \n")