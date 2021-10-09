from torch import optim
import argparse


def create_parser(desctription=''):
    """Function to handle input arguments"""
    parser = argparse.ArgumentParser(description=desctription)

    # Task type and dataset
    task_choices = ['mnist_fmnist', 'splitMNIST', 'permMNIST', 'CIFAR100']
    task_default = 'splitMNIST'
    parser.add_argument('--name', type=str, default=task_default, choices=task_choices)

    # Incremental learning scenario
    scenario_choices = ['task', 'domain', 'class']
    parser.add_argument('--scenario', type=str, default='task', choices=scenario_choices)
    parser.add_argument('--tasks', type=int, default=5, help="Number of tasks")

    # Network type
    models = ['MLP', 'CNN', 'DeepCNN']
    parser.add_argument('--network', type=str, default='MLP', choices=models)
    parser.add_argument('--fc_size', type=int, default=400)
    parser.add_argument('--c', type=int, default=16)

    # Device used to train a network - GPU or CPU
    device_choices = ['cpu', 'gpu']
    parser.add_argument('--device', type=str, default='gpu', choices=device_choices)

    # Number of epochs to train single task
    parser.add_argument('--epochs', type=int, default=10)

    # Optimizer
    optim_choices = ['Adam', 'SGD', 'RMSprop']
    parser.add_argument('--optimizer', type=str, default='Adam', choices=optim_choices)

    # Hyperparameters for optimizers
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.99,
                        help="smoothing constant for RMSprop optimizer")
    parser.add_argument('--betas', nargs='+', type=float, default=(0.9, 0.999),
                        help="coefficients for computing running averages of gradient and its square in ADAM optimizer")

    # Batch sizes
    parser.add_argument('--train_bs', type=int, default=128)
    parser.add_argument('--test_bs', type=int, default=128)

    # EWC params
    parser.add_argument('--lambda_param', type=float, default=1e3)
    parser.add_argument('--fisher_num_batches', type=int, default=None)

    # PackNet params
    parser.add_argument('--prune_perc', type=float, default=0.5)
    parser.add_argument('--retrain_epochs', type=int, default=5)

    # GR params
    parser.add_argument('--latent_dims', type=int, default=100)
    vae_generators = ['MLP', 'CNN', 'DeepCNN']
    parser.add_argument('--generator', type=str, default='MLP', choices=vae_generators)

    # If not specified, this GR params are the same as for main model
    parser.add_argument('--recon_bs', type=float, default=None)
    parser.add_argument('--gen_epochs', type=int, default=None)
    parser.add_argument('--gen_lr', type=float, default=None)
    parser.add_argument('--gen_momentum', type=float, default=None)
    parser.add_argument('--gen_weight_decay', type=float, default=None)
    parser.add_argument('--gen_alpha', type=float, default=None)
    parser.add_argument('--gen_betas', nargs='+', type=float, default=None)

    # Pretraining
    parser.add_argument('--pretrain_cnn', action='store_true', default=False)
    parser.add_argument('--pretrain_cnn_epochs', type=int, default=10)
    parser.add_argument('--pretrain_vae', action='store_true', default=False)
    parser.add_argument('--pretrain_vae_epochs', type=int, default=10)

    # Flag to decide whether train base model without CL to compare
    parser.add_argument('--base_model', action='store_true', default=False, help="train base model to compare")

    # Flags in script compare.py
    parser.add_argument('--ewc', action='store_true', default=False, help="Elastic Weights Consolidation")
    parser.add_argument('--packnet', action='store_true', default=False, help="PackNet algorithm with network pruning")
    parser.add_argument('--gen_rep', action='store_true', default=False, help="Generative Replay")
    # Flag to decide whether to perform joint training to compare
    parser.add_argument('--joint_train', action='store_true', default=False, help="joint training to compare")

    # Flag to print info while program is running
    parser.add_argument('--verbose', action='store_false', default=True, help="print info during program run")
    # Flag to save generated images
    parser.add_argument('--save_imgs', action='store_true', default=False,
                        help="save generated images after every epoch in GR algorithm")
    # Flag to save generated plots
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help="save generated accuracy plots after training")

    return parser


def init_gen_args(args):
    # If parameters not specified, the same as in the main model
    if args.recon_bs is None:
        args.recon_bs = args.train_bs
    if args.gen_epochs is None:
        args.gen_epochs = args.epochs
    if args.gen_lr is None:
        args.gen_lr = args.lr
    if args.gen_momentum is None:
        args.gen_momentum = args.momentum
    if args.gen_weight_decay is None:
        args.gen_weight_decay = args.weight_decay
    if args.gen_alpha is None:
        args.gen_alpha = args.alpha
    if args.gen_betas is None:
        args.gen_betas = args.betas

    return args


def init_optim(model, args, generator=False,):
    if generator:
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), args.gen_lr, args.gen_momentum, args.gen_weight_decay)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), args.gen_lr, args.gen_alpha, 1e-08, args.gen_momentum, args.gen_weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), args.gen_lr, args.gen_betas, 1e-08, args.gen_weight_decay)
    else:
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), args.lr, args.momentum, args.weight_decay)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), args.lr, args.alpha, 1e-08, args.momentum, args.weight_decay)
        else:
            # print(args.betas)
            optimizer = optim.Adam(model.parameters(), args.lr, args.betas, 1e-08, args.weight_decay)

    return optimizer
