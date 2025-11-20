# Imports

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Functions and Classes


## ------------- LDA and QDA (Linear and Quadratic Discriminant Analysis) -------------

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
        norms[:,i] = np.apply_along_axis(weighted_norm,1,test_data-class_means[i,:], cov = class_cov)
        
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

    nb_classes = np.shape(class_means)[0]

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

# Classification by neural network

# Declare a class for MLP (multilayer perceptron)
class MLP_nn(nn.Module):
    
    # class initialization, here we define all the ingredients that we will need for the network (layers, activations...)
    def __init__(self, input_size, hidden1_size, output_size):
        super(MLP_nn, self).__init__()
        # fully connected layer with linear activation
        self.fc0 = nn.Linear(input_size, hidden1_size)
        # fully connected layer with linear activation
        self.fc2 = nn.Linear(hidden1_size, output_size)
        # ReLu activation
        self.relu = nn.ReLU()
        # sigmoid activation
        self.sigmoid = nn.Sigmoid()

        self.L_stack = nn.Sequential(
            self.fc0,
            self.relu,
            self.fc2,
            self.sigmoid
        )
        
    # function to apply the neural network: use all the ingredients above to define the forward pass
    def forward(self, x):
        y_pred = self.L_stack(x)
        return y_pred

def initialize_MLP(input_size, hidden1_size, output_size):
    # create an instance of the MLP_nn class
    MLP = MLP_nn(input_size, hidden1_size, output_size)
    return MLP

# Tensor and scale the data for Pytorch

def scaled_tensorize_data(X_train, y_train, X_val, y_val, X_test, y_test):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)  # y_train est encore une Series

    X_val   = torch.tensor(X_val, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.float32)

    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Create DataLoaders for training and validation

def datasets_and_loaders(X_train, y_train, X_val, y_val, batch_size):
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    return train_loader, val_loader

# Here we define a function to pass arguments and run the training

def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    device,
    epochs=30,
    scheduler_type=None,
    print_every_epochs=1,
    save_best_path=None
    ):
    """
    Trains model and logs metrics every epoch. Prints metrics every `print_every_epochs`.
    scheduler_type:
      - "plateau": ReduceLROnPlateau(optimizer, **scheduler_kwargs)  # expects mode='min'
      - "cosine":  CosineAnnealingLR(optimizer, **scheduler_kwargs)
      - "step":    StepLR(optimizer, **scheduler_kwargs)
      - None:      no scheduler
    """

    # set up learning rate scheduler if specified
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    # Best validation loss for saving the best model
    best_val_loss = np.inf
    epochs_no_improve = 0

    # Here we accumulate the losses over batches to compute the total values over an epoch, and store them


    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = model(Xb).squeeze()
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()

            # accumulate training loss & accuracy
            batch_size = Xb.size(0)
            train_loss_accum += loss.item() * batch_size

            # predictions (binary)
            preds = (outputs >= 0.5).int()
            train_correct += torch.sum(preds == yb.int()).item()
            train_total += batch_size

        # epoch-level train metrics
        avg_train_loss = train_loss_accum / train_total
        train_acc = 100.0 * train_correct / train_total

        # run validation
        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                yv = yv.to(device)
                out_v = model(Xv).squeeze()
                l_v = loss_fn(out_v, yv)
                bs = Xv.size(0)
                val_loss_accum += l_v.item() * bs

                preds_v = (out_v >= 0.5).int()
                val_correct += torch.sum(preds_v == yv.int()).item()
                val_total += bs

        avg_val_loss = val_loss_accum / val_total
        val_acc = 100.0 * val_correct / val_total

        # get current lr (assumes single param_group or want first)
        current_lr = optimizer.param_groups[0]["lr"]

        # log metrics into a dictionary
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau expects a metric (validation loss)
                scheduler.step(avg_val_loss)
            else:
                # continuous schedulers step each epoch
                scheduler.step()

        # save best model if improved on val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            if save_best_path:
                os.makedirs(os.path.dirname(save_best_path), exist_ok=True)
                torch.save(model.state_dict(), save_best_path)
        else:
            epochs_no_improve += 1


        # print periodically (every print_every_epochs)
        if epoch % print_every_epochs == 0 or epoch == 1 or epoch == epochs:
            print(
                f"[Epoch {epoch:3d}/{epochs}] LR {current_lr:.3e} | "
                f"Train loss {avg_train_loss:.4f}  Train acc {train_acc:.2f}% | "
                f"Val loss {avg_val_loss:.4f}  Val acc {val_acc:.2f}%"
            )

    print("Training finished.")
    return history

def print_loss_curve(losses):
    # Print the loss function
    plt.plot(range(len(losses)), losses)
    plt.title('Loss function', size=20)
    plt.xlabel('Epoch', size=20)
    plt.ylabel('Loss value', size=20)

def evaluate_MLP(MLP, X_t_test, y_t_test):
    # compute test accuracy
    y_pred_test = MLP.forward(X_t_test)
    accuracy = torch.sum((y_pred_test.detach().numpy()>0.5) == y_t_test.numpy())/y_t_test.size(0)
    print('Test accuracy: ', accuracy)
