# Imports

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Functions and Classes


# Classification by neural network

# Declare a class for MLP (multilayer perceptron)
class MLP_nn(nn.Module):
    
    # class initialization, here we define all the ingredients that we will need for the network (layers, activations...)
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(MLP_nn, self).__init__()
        # fully connected layer with linear activation
        self.fc0 = nn.Linear(input_size, hidden1_size)
        # fully connected layer with linear activation
        self.fc1 = nn.Linear(hidden1_size, hidden2_size)
        # fully connected layer with linear activation
        self.fc2 = nn.Linear(hidden2_size, output_size)
        # ReLu activation
        self.relu = nn.ReLU()
        # sigmoid activation
        self.sigmoid = nn.Sigmoid()

        self.L_stack = nn.Sequential(
            self.fc0,
            self.relu,
            self.fc1,
            self.relu,
            self.fc2,
            self.sigmoid
        )
        
    # function to apply the neural network: use all the ingredients above to define the forward pass
    def forward(self, x):
        y_pred = self.L_stack(x)
        return y_pred

def initialize_MLP(input_size, hidden1_size, hidden2_size, output_size):
    # create an instance of the MLP_nn class
    MLP = MLP_nn(input_size, hidden1_size, hidden2_size, output_size)
    return MLP

# Here we define a function to pass arguments and run the training

def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    device,
    epochs=30,
    print_every_epochs=1,
    scheduler_type=None,
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
            outputs = model(Xb).squeeze()        # raw logits (binary example)
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()

            # accumulate training loss & accuracy
            batch_size = Xb.size(0)
            train_loss_accum += loss.item() * batch_size

            # predictions (binary)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()
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

                probs_v = torch.sigmoid(out_v)
                preds_v = (probs_v >= 0.5).int()
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
