# ------------------------------------------------------------------------------
#    Functions to evaluate, visualize and save model results
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Size", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        size = list(parameter.shape)
        table.add_row([name, size, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}\n")


def evaluate(model, test_dataloaders, accs_dict, avg_acc_list, avg_loss_list, loss_fn, device, task_nr, verbose=True):
    # Evaluating tasks seen so far
    acc_sum, loss_sum = 0., 0.

    for t_s in range(task_nr + 1):
        # Test model
        if model.scenario == 'class':
            acc, loss = model.test_epoch(test_dataloaders[t_s], loss_fn, device, task_nr=task_nr, verbose=verbose)
            # acc, loss = model.test_epoch(test_dataloaders[t_s], loss_fn, device, task_nr=t_s, verbose=verbose)
        else:
            acc, loss = model.test_epoch(test_dataloaders[t_s], loss_fn, device, task_nr=t_s, verbose=verbose)

        accs_dict['task_' + str(t_s + 1)].append(acc)
        acc_sum += acc
        loss_sum += loss

    # Update average accuracy
    avg_acc_list.append(acc_sum / (task_nr + 1))
    # Update average test loss
    avg_loss_list.append(loss_sum / (task_nr + 1))

    return accs_dict, avg_acc_list, avg_loss_list


def plot_list(items_list, y_desc='', title='', save=False, filename=''):
    # Plot elements from list
    plt.figure()
    for item in items_list:
        if len(item) > 0:
            plt.plot(item)
    plt.xlabel("Epoch"), plt.ylabel(y_desc)
    plt.grid()
    plt.title(title)
    plt.legend(["CL model", "base_model"])

    # Save file
    if save:
        plt.savefig('{}.png'.format(filename), dpi=300)
    # plt.show()


def plot_list_compare(args, items_list, y_desc='', x_desc='Epoch', title='', save=False, filename=''):
    # Plot elements from list
    plt.figure()
    for item in items_list:
        if item is not None:
            plt.plot(item)
    plt.xlabel(x_desc), plt.ylabel(y_desc)
    plt.grid()
    plt.title(title)

    names = []
    if args.base_model:
        names.append('Base')
    if args.ewc:
        names.append('EWC')
    if args.packnet:
        names.append('PackNet')
    if args.gen_rep:
        names.append('GR')
    if args.joint_train:
        names.append('Joint')
    plt.legend(names)

    # Save file
    if save:
        plt.savefig('{}.png'.format(filename), dpi=300)
    # plt.show()


def plot_dict(acc_dict, epochs, tasks, y_desc='', title='', save=False, filename=''):
    # Plot elements from dictionary
    plt.figure()
    for i, (task_key, acc_list) in enumerate(acc_dict.items()):
        x = np.arange(i * epochs, tasks * epochs)
        plt.plot(x, acc_list, label=task_key)
    plt.xlabel("Epoch"), plt.ylabel(y_desc)
    plt.title(title)
    plt.grid(), plt.legend()

    # Save file
    if save:
        plt.savefig('{}.png'.format(filename), dpi=300)
    # plt.show()
