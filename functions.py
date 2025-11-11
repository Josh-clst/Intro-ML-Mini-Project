from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp



## ------------- LDA and QDA (Linear and Quadratic Discriminant Analysis) -------------

# define training/test split

def dataset_split(X):
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33)

# LDA classification algorithm (using Mahalanobis distance)

def weighted_norm(x,cov):    # Mahalanobis distance (assuming x is average subtracted and is a vector)
    return np.dot(x,np.dot(np.linalg.inv(cov),x))

def LDA_classifier_train_cov(training_data,training_labels,nb_classes):
    class_means = np.zeros((nb_classes,training_data.shape[1]))
    class_cov = np.zeros((nb_classes,training_data.shape[1],training_data.shape[1]))
    for i in range(nb_classes):
        class_i_data = training_data[training_labels == i,:]         
        class_means[i,:] = np.mean(class_i_data,axis = 0)  
        class_cov[i,:,:] = np.cov(training_data[training_labels == i,:].T)   # we need to estimate the covariance for each class
    return class_means, class_cov

def LDA_classifier_predict_cov(test_data,class_means,class_cov,nb_classes):
    norms = np.zeros((test_data.shape[0],nb_classes))
    for i in range(nb_classes):
        ## Using apply_along_axis
        norms[:,i] = np.apply_along_axis \ 
            (weighted_norm,1,test_data-class_means[i,:], cov = class_cov)
        
        # ## Not using the apply_along_axis (because I find it simpler for students to understand the correction)
        # ## Note though that it is very long! 2min on my machine
        # # Calculate the difference between test data and class mean
        # diff = test_data - class_means[i,:]
        # # Calculate Mahalanobis distance for each test point
        # for j in range(test_data.shape[0]):
        #     # norms[j,i] = weighted_norm(diff[j,:], cov)
        #     norms[j,i] = np.dot(diff[j,:],np.dot(np.linalg.inv(cov),diff[j,:]))

    predicted_labels = np.argmin(norms,axis =1)
    return predicted_labels

# QDA classification algorithm

def QDA_classifier_train(training_data,training_labels,nb_classes):
    class_means = np.zeros((nb_classes,training_data.shape[1]))
    class_cov = np.zeros((nb_classes,training_data.shape[1],training_data.shape[1]))
    for i in range(nb_classes):
        class_i_data = training_data[training_labels == i,:]
        class_means[i,:] = np.mean(class_i_data,axis = 0)  
        class_cov[i,:,:] = np.cov(training_data[training_labels == i,:].T)   # we need to estimate the covariance for each class
    return class_means, class_cov

def QDA_classifier_predict(test_data,class_means,class_cov,nb_classes):
    discriminative_functions = np.zeros((test_data.shape[0],nb_classes))
    for i in range(nb_classes):
        discriminative_functions[:,i] = -np.apply_along_axis \
            (weighted_norm,1,test_data-class_means[i,:], cov = class_cov[i,:,:]) - test_data.shape[1]/2*np.log(2*np.pi) \
            -1/2*np.log(np.linalg.det(class_cov[i,:,:]))
    predicted_labels = np.argmax(discriminative_functions,axis =1)
    return predicted_labels

# train and test accuracy

def train_test_accuracy_cov(X_train,y_train,X_test,y_test, class_means, cov, classifier):
    nb_classes = np.shape(class_means)[0]
    predicted_labels_train = classifier(X_train,class_means,cov,nb_classes)
    predicted_labels_test = classifier(X_test,class_means,cov,nb_classes)
    training_accuracy = (np.sum((y_train-predicted_labels_train) == 0)/np.shape(y_train))[0]
    test_accuracy = (np.sum((y_test-predicted_labels_test) == 0)/np.shape(y_test))[0]
    return training_accuracy, test_accuracy

# plot functions

def plot_decision_boundary_cov(x,y,X,labels,class_means, cov, classifier):
    # to be completed
    # x and y are the vectors giving the extent of the 2D grid
    # X is the data
    # labels are the classification labels
    # classifier is any trained classifier that can be used for prediction
    # nb_classes = np.shape(class_means)[0]
    # [xx,yy] = np.meshgrid(x,y)
    # Z = classifier(np.c_[xx.ravel(), yy.ravel()],class_means,cov,nb_classes)
    # Z = Z.reshape(grid_size,grid_size)
    # plt.scatter(X[:,0],X[:,1], c = labels)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha = 0.6)

    x_coords, y_coords = np.meshgrid(x, y)
    
    # Convert the 2D grids into a list of coordinate pairs
    all_grid_points = np.column_stack([x_coords.flatten(), y_coords.flatten()])
    
    # Classify all points at once (each coordinate pair is a point)
    classify_grid_flat = classifier(all_grid_points, class_means, cov, nb_classes)
    
    # Reshape back to 2D grid for plotting
    classify_grid = classify_grid_flat.reshape(x_coords.shape)

    # Plot the results
    plt.scatter(X[:,0],X[:,1], c = labels)
    plt.contourf(x_coords, y_coords, classify_grid, cmap=plt.cm.RdBu, alpha = 0.6)
    