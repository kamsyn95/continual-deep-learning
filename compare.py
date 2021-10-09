import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy

from models.base_nets import CNN, MLP, DeepCNN
from models.pretrain import pretrain_cnn, pretrain_vae

from models.EWC_method import EWCMultitask
from models.GenReplayVAE import GenerativeReplay
from models.prune import Pruner, init_masks

from data.load import get_multitask_experiment
from data.utils import load_mnist_fmnist, CatDataloaders
from eval_utils import evaluate, plot_list_compare, count_parameters

from options import create_parser, init_optim, init_gen_args


# Command line Input arguments
parser = create_parser()
args = parser.parse_args()

if args.device == 'gpu':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')


# Create multitask datasets
if args.name == 'permMNIST':
    args.scenario = 'domain'
if args.name == 'mnist_fmnist':
    args.scenario = 'task'
    args.tasks = 2
    (train_datasets, test_datasets), config, classes_per_task = load_mnist_fmnist()
else:
    (train_datasets, test_datasets), config, classes_per_task = \
        get_multitask_experiment(name=args.name, scenario=args.scenario, tasks=args.tasks,
                                 verbose=args.verbose, mnist28=False)
print("Data loaded")

multihead = True if args.scenario == 'task' else False

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


# define model
if args.network == 'CNN':
    model = CNN(in_size=config['size'], in_channels=config['channels'], c=args.c, fc_size=args.fc_size,
                classes=classes_per_task, tasks=args.tasks, scenario=args.scenario, multihead=multihead)
elif args.network == 'DeepCNN':
    model = DeepCNN(in_size=config['size'], in_channels=config['channels'], c=args.c, fc_size=args.fc_size,
                    classes=classes_per_task, tasks=args.tasks, scenario=args.scenario, multihead=multihead)
else:
    model = MLP(in_size=config['size'], in_channels=config['channels'], fc_size=args.fc_size,
                classes=classes_per_task, tasks=args.tasks, scenario=args.scenario, multihead=multihead)

# Define optimizer
model.to(device)
model.optimizer = init_optim(model, args)

# Visualize model's layers
if args.verbose:
    count_parameters(model)


# Pretraining on CIFAR-10 if we use CIFAR-100 as dataset for CL experiments
if args.pretrain_cnn and args.name == 'CIFAR100':
    model = pretrain_cnn(model, device, bs=args.train_bs, epochs=args.pretrain_cnn_epochs)
    # Reset optimizers
    model.optimizer = init_optim(model, args)

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# List to store average accuracy over all tasks seen so far
avg_acc_base, avg_acc_ewc, avg_acc_pn, avg_acc_gr, avg_acc_joint = None, None, None, None, None



####################### BASE MODEL ##############################
if args.base_model:
    print('\n' + ('-' * 16) + ' BASE MODEL ' + ('-' * 16) + '\n')

    # Copy original model
    base_model = deepcopy(model)
    # Define optimizer
    base_model.to(device)
    base_model.optimizer = init_optim(base_model, args)

    # Dictionary and lists to store accuracies and losses during training
    accs_base, avg_acc_base, avg_test_loss_base = dict(), [], []

    # Sequential training for each task
    for t in range(args.tasks):
        print("\n----------- TASK {} ------------".format(t + 1))
        accs_base['task_' + str(t + 1)] = []

        for i in range(args.epochs):
            print(f"Epoch {i + 1}\n-------------------------------")
            # Train base model
            base_model.train_epoch(train_dataloaders[t], loss_fn, device, task_nr=t)
            # Test base model
            print("\n-----Base model TEST Accuracies-----")
            accs_base, avg_acc_base, avg_test_loss_base = \
                evaluate(base_model, test_dataloaders, accs_base, avg_acc_base, avg_test_loss_base,
                         loss_fn, device, task_nr=t, verbose=args.verbose)

        # Reset optimizer
        base_model.optimizer = init_optim(base_model, args)

    print('\n' + ('-' * 16) + ' END OF BASE MODEL TRAINING ' + ('-' * 16) + '\n')



####################### ELASTIC WEIGHT CONSOLIDATION (EWC) ##############################
if args.ewc:
    print('\n' + ('-' * 16) + ' ELASTIC WEIGHT CONSOLIDATION (EWC) ' + ('-' * 16) + '\n')

    # Copy original model
    ewc_model = deepcopy(model)
    # Define optimizer
    ewc_model.to(device)
    ewc_model.optimizer = init_optim(ewc_model, args)

    # Initialize EWC class
    ewc = EWCMultitask(model=ewc_model, crit=loss_fn, weight=args.lambda_param)

    # Dictionary and lists to store accuracies and losses during training
    accs_ewc, avg_acc_ewc, avg_test_loss_ewc = dict(), [], []

    # Sequential training for each task
    for t in range(args.tasks):
        print("\n----------- TASK {} ------------".format(t + 1))
        accs_ewc['task_' + str(t + 1)] = []

        for i in range(args.epochs):
            print(f"Epoch {i + 1}\n-------------------------------")
            # Train EWC model
            ewc.train_epoch(train_dataloaders[t], device, task_nr=t)
            # Test EWC model
            print("\n-----EWC model TEST Accuracies-----")
            accs_ewc, avg_acc_ewc, avg_test_loss_ewc = evaluate(ewc.model, test_dataloaders,
                                                                accs_ewc, avg_acc_ewc, avg_test_loss_ewc,
                                                                loss_fn, device, task_nr=t, verbose=args.verbose)

        # If we reach last task break the loop and don't compute FIM
        if t == args.tasks - 1:
            break

        # Compute regularization terms - mean and FIM for every layer in network
        ewc.compute_fim(task_name=args.name, dataset=train_datasets[t], batch_size=64, device=device,
                        task_nr=t, num_batches=args.fisher_num_batches)

        # Reset optimizer
        ewc.model.optimizer = init_optim(ewc.model, args)

    print('\n' + ('-' * 16) + ' END OF EWC TRAINING ' + ('-' * 16) + '\n')



################################## PACKNET ##############################################
if args.packnet:
    print('\n' + ('-' * 16) + ' PACKNET ' + ('-' * 16) + '\n')

    # Copy original model
    packnet_model = deepcopy(model)
    # Define optimizer
    packnet_model.to(device)
    packnet_model.optimizer = init_optim(packnet_model, args)

    # Create Pruner
    masks = init_masks(packnet_model)
    pruner = Pruner(packnet_model, prune_perc=args.prune_perc, previous_masks=masks,
                    train_bias=False, train_bn=False, device=device)

    # Dictionary and lists to store accuracies and losses during training
    accs_pn, avg_acc_pn, avg_test_loss_pn = dict(), [], []


    # Sequential training for each task
    for t in range(args.tasks):
        print("\n----------- TASK {} ------------".format(t + 1))
        accs_pn['task_' + str(t + 1)] = []

        for i in range(args.epochs):
            print(f"Epoch {i + 1}\n-------------------------------")
            # Train pruned model
            pruner.train_epoch(train_dataloaders[t], loss_fn, device, task_nr=t)

            # Evaluating tasks seen so far
            print("\n-----PACKNET model TEST Accuracies-----")
            acc_sum, loss_sum = 0., 0.
            for t_s in range(t + 1):
                # Test pruned model
                puner_copy = deepcopy(pruner)
                # Copy is needed - weights with mask_idx > dataset_idx are set to '0'
                puner_copy.apply_mask(dataset_idx=t_s+1)
                acc, loss = puner_copy.model.test_epoch(test_dataloaders[t_s], loss_fn, device, task_nr=t_s)
                accs_pn['task_' + str(t_s + 1)].append(acc)
                acc_sum += acc
                loss_sum += loss

            # Update average accuracy
            avg_acc_pn.append(acc_sum / (t + 1))
            # Update average test loss
            avg_test_loss_pn.append(loss_sum / (t + 1))

        # Perform pruning
        pruner.prune()
        pruner.make_pruned_zero()

        # Retrain pruned model
        for i in range(args.retrain_epochs):
            print(f"Retraining pruned model, Epoch {i + 1}\n-------------------------------")
            pruner.train_epoch(train_dataloaders[t], loss_fn, device, task_nr=t)
            puner_copy = deepcopy(pruner)
            puner_copy.apply_mask(dataset_idx=t+1)
            puner_copy.model.test_epoch(test_dataloaders[t], loss_fn, device, task_nr=t)
        print("Prunig after TASK %d ended" % t)

        # Make pruned params available for new dataset.
        pruner.make_finetuning_mask()

        # Reset optimizer
        pruner.model.optimizer = init_optim(pruner.model, args)

    print('\n' + ('-' * 16) + ' END OF PACKNET TRAINING' + ('-' * 16) + '\n')



####################### GENERATIVE REPLAY (GR) ##############################
if args.gen_rep:
    print('\n' + ('-' * 16) + ' GENERATIVE REPLAY (GR)  ' + ('-' * 16) + '\n')

    # Set parameters for generator
    args = init_gen_args(args)
    gen_separate_train = True if args.epochs != args.gen_epochs else False

    # Initialize GR class
    GR = GenerativeReplay(net_type=args.network, vae_type=args.generator, in_size=config['size'], in_channels=config['channels'],
                          vae_latent_dims=args.latent_dims, c=args.c, fc_size=args.fc_size,
                          classes=classes_per_task, tasks=args.tasks, scenario=args.scenario, multihead=multihead)

    # Copy model
    GR.model = deepcopy(model)

    # Set optimizers
    GR.to(device)
    GR.model.optimizer = init_optim(GR.model, args)
    GR.vae.optimizer = init_optim(GR.vae, args, generator=True)

    if args.verbose:
        print(('-' * 16) + ' Generator layers ' + ('-' * 16))
        count_parameters(GR.vae)

    # Pretraining VAE
    if args.name == 'CIFAR100' and args.pretrain_vae:
        GR.vae = pretrain_vae(GR.vae, device, bs=args.train_bs, epochs=args.pretrain_vae_epochs)
        GR.vae.optimizer = init_optim(GR.vae, args, generator=True)  # Reset optimizer

    # Dictionary and lists to store accuracies and losses during training
    accs_gr, avg_acc_gr, avg_test_loss_gr = dict(), [], []

    # New list of dataloaders to train on the next task
    dls_list = []


    # Sequential training for each task
    for t in range(args.tasks):
        print("----------- TASK {} ------------".format(t + 1))
        # Concat current dl with generated dl
        dls_list.append(train_dataloaders[t])
        DL_concated = CatDataloaders(dls_list)

        # List to store accuracy for current task
        accs_gr['task_' + str(t + 1)] = []

        # Training loop for both model and generator or only for model
        for i in range(args.epochs):
            print(f"Epoch {i + 1}\n-------------------------------")

            # Train model with both current and generated data
            if gen_separate_train:
                GR.train_epoch_model(DL_concated, loss_fn, device, task_nr=t, verbose=args.verbose)
            else:
                GR.train_epoch(DL_concated, loss_fn, device, task_nr=t, epoch=i, verbose=args.verbose, save=args.save_imgs)

            # Evaluating tasks seen so far
            print("\n-----GR model TEST Accuracies-----")
            accs_gr, avg_acc_gr, avg_test_loss_gr = evaluate(GR.model, test_dataloaders, accs_gr, avg_acc_gr, avg_test_loss_gr,
                                                             loss_fn, device, task_nr=t, verbose=args.verbose)

        # If we reach last task break the loop and don't train generator
        if t == args.tasks - 1:
            break

        # Separate train of generator if number of generator's train epochs differ from model's train epochs
        if gen_separate_train:
            for i in range(args.gen_epochs):
                GR.train_epoch_generator(DL_concated, device, task_nr=t, epoch=i, verbose=args.verbose, save=args.save_imgs)

        # New list of dataloaders to train on the next task
        dls_list = []

        recon_batch_size = int(args.recon_bs / (t + 1))
        ds_size = int(len(train_dataloaders[t].dataset) / (t + 1))
        for r_s in range(t + 1):
            gen_dl = GR.generate_dataset(ds_size, device, task_nr=r_s, batch_size=recon_batch_size)
            dls_list.append(gen_dl)

        # RESET optimizers
        GR.model.optimizer = init_optim(GR.model, args)
        GR.vae.optimizer = init_optim(GR.vae, args, generator=True)

    print('\n' + ('-' * 16) + ' END OF GENERATIVE REPLAY TRAINING' + ('-' * 16) + '\n')



####################### JOINT TRAINING ##############################
if args.joint_train:
    print('\n' + ('-' * 16) + ' JOINT TRAINING ' + ('-' * 16) + '\n')
    # Copy model
    joint_model = deepcopy(model)
    # Define optimizer
    joint_model.to(device)
    joint_model.optimizer = init_optim(joint_model, args)

    # Dictionary and lists to store accuracies and losses during training
    accs_joint, avg_acc_joint, avg_test_loss_joint = dict(), [], []
    # New list of dataloaders to train on the next task
    dls_list = []

    # Sequential training for each task
    for t in range(args.tasks):
        print("\n----------- TASK {} ------------".format(t + 1))
        accs_joint['task_' + str(t + 1)] = []

        # Concat current dl with previous dls
        dls_list.append(train_dataloaders[t])
        DL_concated = CatDataloaders(dls_list)

        for i in range(args.epochs):
            print(f"Epoch {i + 1}\n-------------------------------")
            # Train joint model
            joint_model.train_epoch_joint(DL_concated, loss_fn, device, verbose=args.verbose)

            # Test joint model
            print("\n-----Joint training TEST Accuracies-----")
            accs_joint, avg_acc_joint, avg_test_loss_joint = \
                evaluate(joint_model, test_dataloaders, accs_joint, avg_acc_joint, avg_test_loss_joint,
                         loss_fn, device, task_nr=t, verbose=args.verbose)

        # Reset optimizer
        joint_model.optimizer = init_optim(joint_model, args)

    print('\n' + ('-' * 16) + ' END OF JOINT TRAINING' + ('-' * 16) + '\n')



# Visualize results
# Plot average accuracy and Save plot
filename = "store/plots/ALL_{}_{}_{}tasks_{}_fc{}_lr{}_bs{}_epochs{}_avg".format(
    args.name, args.scenario, args.tasks, args.network, args.fc_size, args.lr, args.train_bs, args.epochs)

# Plot test accuracies for every epoch in each task
plot_list_compare(args, [avg_acc_base, avg_acc_ewc, avg_acc_pn, avg_acc_gr, avg_acc_joint],
                  y_desc='Average accuracy', title='', save=args.save_plots, filename=filename)


# Plot accuracies at the end of each task
avg_acc_base_end = [avg_acc_base[((t + 1)*args.epochs) - 1] for t in range(args.tasks)] if args.base_model else None
avg_acc_ewc_end = [avg_acc_ewc[((t + 1)*args.epochs) - 1] for t in range(args.tasks)] if args.ewc else None
avg_acc_pn_end = [avg_acc_pn[((t + 1)*args.epochs) - 1] for t in range(args.tasks)] if args.packnet else None
avg_acc_gr_end = [avg_acc_gr[((t + 1)*args.epochs) - 1] for t in range(args.tasks)] if args.gen_rep else None
avg_acc_joint_end = [avg_acc_joint[((t + 1)*args.epochs) - 1] for t in range(args.tasks)] if args.joint_train else None

# Plot test accuracies at the end of each task
plot_list_compare(args, [avg_acc_base_end, avg_acc_ewc_end, avg_acc_pn_end, avg_acc_gr_end, avg_acc_joint_end],
                  y_desc='Average accuracy', x_desc='Task', title='', save=args.save_plots, filename=filename+'_task_end')

plt.show()
