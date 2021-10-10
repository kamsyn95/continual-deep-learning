import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy

from models.GenReplayVAE import GenerativeReplay
from models.pretrain import pretrain_cnn, pretrain_vae

from data.load import get_multitask_experiment
from data.utils import CatDataloaders, load_mnist_fmnist
from eval_utils import evaluate, plot_list, plot_dict, count_parameters

from options import create_parser, init_optim, init_gen_args


# Command line Input arguments
parser = create_parser()
args = parser.parse_args()

# Set parameters for generator
args = init_gen_args(args)
gen_separate_train = True if args.epochs != args.gen_epochs else False

# Set device to train and test a network
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


# args.tasks = 5

# define model
GR = GenerativeReplay(
    net_type=args.network,
    vae_type=args.generator,
    in_size=config["size"],
    in_channels=config["channels"],
    vae_latent_dims=args.latent_dims,
    c=args.c,
    fc_size=args.fc_size,
    classes=classes_per_task,
    tasks=args.tasks,
    scenario=args.scenario,
    multihead=multihead,
)

# Set optimizers
GR.to(device)
GR.model.optimizer = init_optim(GR.model, args)
GR.vae.optimizer = init_optim(GR.vae, args, generator=True)

print(("-" * 16) + " Main model layers " + ("-" * 16))
count_parameters(GR.model)
print(("-" * 16) + " Generator layers " + ("-" * 16))
count_parameters(GR.vae)


# Pretraining on CIFAR-10 if we use CIFAR-100 as dataset for CL experiments
if args.name == "CIFAR100":
    # Pretraining CNN
    if args.pretrain_cnn:
        GR.model = pretrain_cnn(GR.model, device, bs=args.train_bs, epochs=args.pretrain_cnn_epochs)
        GR.model.optimizer = init_optim(GR.model, args)  # Reset optimizer
    # Pretraining VAE
    if args.pretrain_vae:
        GR.vae = pretrain_vae(GR.vae, device, bs=args.train_bs, epochs=args.pretrain_vae_epochs)
        GR.vae.optimizer = init_optim(GR.vae, args, generator=True)  # Reset optimizer


# Copy model without CL capabilities to train it as baseline
if args.base_model:
    base_model = deepcopy(GR.model)
    base_model.to(device)
    base_model.optimizer = init_optim(base_model, args)


# Dataloader
train_dataloaders = []
test_dataloaders = []
for t in range(args.tasks):
    # Training DLs
    dl_train = DataLoader(train_datasets[t], batch_size=args.train_bs, shuffle=True)
    train_dataloaders.append(dl_train)
    # Testing DLs
    dl_test = DataLoader(test_datasets[t], batch_size=args.test_bs, shuffle=True)
    test_dataloaders.append(dl_test)

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Dictionary to store accuracy for each task
accs, accs_base = dict(), dict()
# Average accuracy over all task seen so far
avg_acc, avg_acc_base = [], []
# Average test losses
avg_test_loss, avg_test_loss_base = [], []
# Train losses
train_losses, train_losses_base = [], []

# New list of dataloaders to train on the next task
dls_list = []


# Sequential training for each task
for t in range(args.tasks):
    print("----------- TASK {} ------------".format(t + 1))
    # Concat current dl with generated dl
    dls_list.append(train_dataloaders[t])
    DL_concated = CatDataloaders(dls_list)

    # List to store accuracy for current task
    accs["task_" + str(t + 1)] = []
    accs_base["task_" + str(t + 1)] = []

    # Training loop for both model and generator or only for model
    for i in range(args.epochs):
        print(f"Epoch {i + 1}\n-------------------------------")

        # Train model with both current and generated data
        if gen_separate_train:
            train_loss_GR_model = GR.train_epoch_model(DL_concated, loss_fn, device, task_nr=t, verbose=args.verbose)
        else:
            train_loss_GR_model, _ = GR.train_epoch(
                DL_concated, loss_fn, device, task_nr=t, epoch=i, verbose=args.verbose, save=args.save_imgs
            )
        train_losses.append(train_loss_GR_model)
        # Evaluating tasks seen so far
        print("\n-----GR model TEST Accuracies-----")
        accs, avg_acc, avg_test_loss = evaluate(
            GR.model, test_dataloaders, accs, avg_acc, avg_test_loss, loss_fn, device, task_nr=t, verbose=args.verbose
        )

        # Train base model
        if args.base_model:
            train_loss_base = base_model.train_epoch(train_dataloaders[t], loss_fn, device, task_nr=t)
            train_losses_base.append(train_loss_base)
            # Test base model
            print("\n-----BASE model TEST Accuracies-----")
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

    # If we reach last task break the loop and don't train generator
    if t == args.tasks - 1:
        break

    # Separate train of generator if number of generator's train epochs differ from model's train epochs
    if gen_separate_train:
        for i in range(args.gen_epochs):
            GR.train_epoch_generator(
                DL_concated, device, task_nr=t, epoch=i, verbose=args.verbose, save=args.save_imgs
            )

    # New list of dataloaders to train on the next task
    dls_list = []

    # # After training on current task Generate new dataset from VAE
    # prev_device = device
    # device = torch.device('cpu')  # Out of memory on cuda
    # GR.to(device)

    recon_batch_size = int(args.recon_bs / (t + 1))
    ds_size = int(len(train_dataloaders[t].dataset) / (t + 1))
    for r_s in range(t + 1):
        gen_dl = GR.generate_dataset(ds_size, device, task_nr=r_s, batch_size=recon_batch_size)
        dls_list.append(gen_dl)

    # # Changing device to previous device
    # device = prev_device
    # GR.to(device)

    # RESET optimizers
    GR.model.optimizer = init_optim(GR.model, args)
    GR.vae.optimizer = init_optim(GR.vae, args, generator=True)

print("----------- END OF TRAINING ------------")


# Visualize results
# Plot average accuracy and Save plot
filename = "store/plots/GR/VAE_{}_{}_{}_ld{}_lr{}_fc{}_tbs{}_epochs{}_{}_average".format(
    args.name,
    args.scenario,
    GR.model._get_name(),
    args.latent_dims,
    args.lr,
    args.fc_size,
    args.train_bs,
    args.epochs,
    args.gen_epochs,
)

plot_list([avg_acc, avg_acc_base], y_desc="Average accuracy", title="", save=args.save_plots, filename=filename)

# Plot train losses
plot_list([train_losses, train_losses_base], y_desc="Train loss", save=False)

# Plot test losses
plot_list([avg_test_loss, avg_test_loss_base], y_desc="Test loss", save=False)


# Plot accs for each task and Save plot
filename = "store/plots/GR/VAE_{}_{}_{}_ld{}_lr{}_fc{}_tbs{}_epochs{}_{}".format(
    args.name,
    args.scenario,
    GR.model._get_name(),
    args.latent_dims,
    args.lr,
    args.fc_size,
    args.train_bs,
    args.epochs,
    args.gen_epochs,
)

plot_dict(accs, args.epochs, args.tasks, y_desc="Accuracy", title="", save=args.save_plots, filename=filename)

# Plot accs for each task for BASE MODEL
if args.base_model:
    plot_dict(accs_base, args.epochs, args.tasks, y_desc="Base Accuracy", title="", save=False)

plt.show()
