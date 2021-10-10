import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from copy import deepcopy

from models.base_nets import CNN, MLP, DeepCNN
from models.EWC_method import EWCMultitask
from models.pretrain import pretrain_cnn

from data.load import get_multitask_experiment
from data.utils import load_mnist_fmnist
from eval_utils import evaluate, plot_list, plot_dict, count_parameters

from options import create_parser, init_optim


# Command line Input arguments
parser = create_parser()
args = parser.parse_args()

if args.device == "gpu":
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")


# Create multitask datasets
if args.name == "permMNIST":
    args.scenario = "domain"
if args.name == "mnist_fmnist":
    args.scenario = "task"
    args.tasks = 2
    (train_datasets, test_datasets), config, classes_per_task = load_mnist_fmnist()
else:
    (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
        name=args.name, scenario=args.scenario, tasks=args.tasks, verbose=args.verbose, mnist28=False
    )
print("Data loaded")

multihead = True if args.scenario == "task" else False


# define model
if args.network == "CNN":
    model = CNN(
        in_size=config["size"],
        in_channels=config["channels"],
        c=args.c,
        fc_size=args.fc_size,
        classes=classes_per_task,
        tasks=args.tasks,
        scenario=args.scenario,
        multihead=multihead,
    )
elif args.network == "DeepCNN":
    model = DeepCNN(
        in_size=config["size"],
        in_channels=config["channels"],
        c=args.c,
        fc_size=args.fc_size,
        classes=classes_per_task,
        tasks=args.tasks,
        scenario=args.scenario,
        multihead=multihead,
    )
else:
    model = MLP(
        in_size=config["size"],
        in_channels=config["channels"],
        fc_size=args.fc_size,
        classes=classes_per_task,
        tasks=args.tasks,
        scenario=args.scenario,
        multihead=multihead,
    )

# Define optimizer
model.to(device)
model.optimizer = init_optim(model, args)

# Visualize model's layers
count_parameters(model)


# Pretraining on CIFAR-10 if we use CIFAR-100 as dataset for CL experiments
if args.pretrain_cnn and args.name == "CIFAR100":
    model = pretrain_cnn(model, device, bs=args.train_bs, epochs=args.pretrain_cnn_epochs)
    # Reset optimizers
    model.optimizer = init_optim(model, args)

# Copy model without CL capabilities to train it as baseline
if args.base_model:
    base_model = deepcopy(model)
    base_model.to(device)
    base_model.optimizer = init_optim(base_model, args)


# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Initialize EWC class
ewc = EWCMultitask(model=model, crit=loss_fn, weight=args.lambda_param)

# Dataloader
train_dataloaders = []
test_dataloaders = []
for t in range(args.tasks):
    # Training DLs
    dl_train = DataLoader(train_datasets[t], batch_size=args.train_bs, shuffle=True)
    train_dataloaders.append(dl_train)
    # Testing DLs
    dl_test = DataLoader(test_datasets[t], batch_size=args.test_bs, shuffle=False)
    test_dataloaders.append(dl_test)


# Dictionary to store accuracy for each task
accs, accs_base = dict(), dict()
# Average accuracy over all task seen so far
avg_acc, avg_acc_base = [], []
# Average test losses
avg_test_loss, avg_test_loss_base = [], []
# Train losses
train_losses, train_losses_base = [], []


# Sequential training for each task
for t in range(args.tasks):
    print("\n----------- TASK {} ------------".format(t + 1))
    # List to store accuracy for current task
    accs["task_" + str(t + 1)] = []
    accs_base["task_" + str(t + 1)] = []

    for i in range(args.epochs):
        print(f"Epoch {i + 1}\n-------------------------------")

        # Train EWC model
        train_loss_ewc = ewc.train_epoch(train_dataloaders[t], device, task_nr=t)
        train_losses.append(train_loss_ewc)
        # Test EWC model
        print("\n-----EWC model TEST Accuracies-----")
        accs, avg_acc, avg_test_loss = evaluate(
            ewc.model, test_dataloaders, accs, avg_acc, avg_test_loss, loss_fn, device, task_nr=t, verbose=args.verbose
        )

        # Train base model
        if args.base_model:
            train_loss_base = base_model.train_epoch(train_dataloaders[t], loss_fn, device, task_nr=t)
            train_losses_base.append(train_loss_base)
            # Test base model
            print("\n-----Base model TEST Accuracies-----")
            accs_base, avg_acc_base, avg_test_loss_base = evaluate(
                base_model,
                test_dataloaders,
                accs_base,
                avg_acc_base,
                avg_test_loss_base,
                loss_fn,
                device,
                task_nr=t,
                verbose=args.verbose,
            )
            # Reset optimizer
            base_model.optimizer = init_optim(base_model, args)

    # If we reach last task break the loop and don't compute FIM
    if t == args.tasks - 1:
        break

    # Compute regularization terms - mean and FIM for every layer in network
    ewc.compute_fim(
        task_name=args.name,
        dataset=train_datasets[t],
        batch_size=64,
        device=device,
        task_nr=t,
        num_batches=args.fisher_num_batches,
    )

    # Reset optimizer
    ewc.model.optimizer = init_optim(ewc.model, args)

print("----------- END OF TRAINING ------------")


# Visualize results
# Plot average accuracy and Save plot
filename = "store/plots/EWC/{}_{}_lam{}_lr{}_bs{}_epochs{}_avg_bn_pretrain{}".format(
    args.name, model._get_name(), args.lambda_param, args.lr, args.train_bs, args.epochs, args.pretrain_cnn_epochs
)

plot_list([avg_acc, avg_acc_base], y_desc="Average accuracy", title="", save=args.save_plots, filename=filename)

# Plot train losses
plot_list([train_losses, train_losses_base], y_desc="Train loss", save=False)

# Plot test losses
plot_list([avg_test_loss, avg_test_loss_base], y_desc="Test loss", save=False)


# Plot accs for each task and Save plot
filename = "store/plots/EWC/{}_{}_lam{}_lr{}_bs{}_epochs{}_bn_pretrain{}".format(
    args.name, model._get_name(), args.lambda_param, args.lr, args.train_bs, args.epochs, args.pretrain_cnn_epochs
)

plot_dict(accs, args.epochs, args.tasks, y_desc="Accuracy", title="", save=args.save_plots, filename=filename)

# Plot accs for each task for BASE MODEL
if args.base_model:
    plot_dict(accs_base, args.epochs, args.tasks, y_desc="Base Accuracy", title="", save=False)

plt.show()
