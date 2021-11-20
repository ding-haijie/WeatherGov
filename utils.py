import os
import random
import logging
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.utils.data as Data


def save_checkpoint(experiment_time, model, optimizer):
    check_file_exist('./results/checkpoints')
    checkpoint_path = './results/checkpoints/' + experiment_time + '.pth'
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(latest, device, file_name=None):
    """ load the latest checkpoint """
    checkpoints_dir = './results/checkpoints'
    if latest:
        file_list = os.listdir(checkpoints_dir)
        file_list.sort(key=lambda fn: os.path.getmtime(
            checkpoints_dir + '/' + fn))
        checkpoint = torch.load(checkpoints_dir + '/' + file_list[-1], map_location=device)
    else:
        if file_name is None:
            raise ValueError('checkpoint_path cannot be empty!')
        checkpoint = torch.load(checkpoints_dir + '/' + file_name)
    return checkpoint


def fix_seed(seed):
    """ set random seed to ensure reproducibility """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def check_file_exist(file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)


def get_data_loader(data, batch_size, device, tag="train"):
    seq_info_numpy, seq_target_numpy = data.process_seq(tag)
    _dataset = Data.TensorDataset(
        torch.tensor(seq_info_numpy, dtype=torch.float, device=device),
        torch.tensor(seq_target_numpy, dtype=torch.float, device=device))
    data_loader = Data.DataLoader(
        dataset=_dataset,
        batch_size=batch_size,
        shuffle=(tag == "train"))
    return data_loader


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(levelname)s] %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def record_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_min = int(elapsed_time / 60)
    elapsed_sec = int(elapsed_time - (elapsed_min * 60))
    return elapsed_min, elapsed_sec


def show_plot(plot_loss_train, plot_loss_dev):
    plt.figure()
    plt.plot(plot_loss_train)
    plt.plot(plot_loss_dev)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.show()


def random_list(start: int, stop: int, length: int):
    # start, stop, length must be int type
    rand_list = []
    while len(rand_list) < length:
        rand_number = random.randint(start, stop)
        if rand_number not in rand_list:
            rand_list.append(rand_number)
    return rand_list
