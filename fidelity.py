import argparse
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from modules.dataset_loader import image_data  # data transform
from modules.loss_function import NT_Xent  # loss function
from modules.TSFE_model import TSFE_model  # TSFE
from modules.encoder import get_network  # encoder
from train import train
import time
import numpy as np
import pandas as pd
import torch
import random
import os
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")


# distribution training parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0,5,6,7'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(local_rank, args):
    """
    local_rank: the number at the time of process creation, assuming 2 machines with 8 GPUs per machine, then
                there are a total of 16 processes, rank ∈ [0,15]. local_ Rank is the serial number of the processes on each machine.
                there are 0,1,2,3,4,5,6,7 on machine one, and 0,1,2,3,4,5,6,7 on machine two
                therefore, loca_ Rank belongs to [0,7]
    args：Parameters passed to each process          
    """

    if args.nprocs > 1:  # use distribution training
        torch.distributed.init_process_group(
            backend="nccl", rank=local_rank, world_size=args.world_size)
        torch.cuda.set_device(local_rank)
        # update parameters
        args.device = torch.device("cuda:{}".format(local_rank))
        args.batch_size = int(args.batch_size / args.nprocs)
        print(f"Training BATCH_SIZE:{args.batch_size}")

    # seed
    seed_everything(args.seed)

    train_list = []

    # distorted images
    for i in range(1, 4745):  # 4744*5*4
        # image id is a number of five digits
        img_id = ('00000' + str(i))[-5:]
        for dist in ['gblur', 'gnoise', 'jpg', 'jp2k']:  # 4 types of distortion
            for level in range(1, 6):  # 5 levels
                # path is ./exploration_database_and_code/distort_images/gblur/[dist_type]_[image_ID]_[dist_level]
                path = args.distorted_image_path_pre + '/' + dist + '/' + dist + \
                    '_' + img_id + '_' + str(level)
                path += '.bmp' if dist in ['gblur', 'gnoise'] else '.jpg'
                train_list.append(path)
    # original image
    for i in range(1, 4745):
        img_id = ('00000' + str(i))[-5:]
        path = args.pristine_image_path_pre + '/' + img_id + '.bmp'
        train_list.append(path)

    # loader for synthetic distortions data
    train_dataset_syn = image_data(train_list, image_size=args.img_size)

    if args.nprocs > 1:
        train_sampler_syn = torch.utils.data.distributed.DistributedSampler(
            train_dataset_syn)
    else:
        train_sampler_syn = None

    train_loader_syn = torch.utils.data.DataLoader(
        train_dataset_syn,
        batch_size=args.batch_size,
        shuffle=(train_sampler_syn is None),
        num_workers=args.workers,
        drop_last=True,
        sampler=train_sampler_syn,
    )

    # initialize ResNet
    encoder = get_network(args.encoder_network, pretrained=False)
    args.n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = TSFE_model(encoder, args.n_features)

    # sgd optmizer
    optimizer_tsfe = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9)

    # loss
    criterion = NT_Xent(args.batch_size, args.temperature,
                        args.device)

    # SyncBN
    if args.nprocs > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])

    model = model.cuda()

    epoch_losses = []
    for epoch in range(args.epochs):
        # shuffle for distribution training
        train_sampler_syn.set_epoch(epoch)

        start = time.time()
        # train one epoch (for the whole training set)
        loss_epoch = train(local_rank, train_loader_syn, model, criterion, optimizer_tsfe)
        end = time.time()

        if local_rank == 0:
            print("time:{}".format(np.round(end-start, 4)))
            # record loss in each epoch
            epoch_losses.append(loss_epoch)
            # print loss_epoch
            print(f"Epoch : {epoch+1} - loss : {loss_epoch:.4f} \n")

        if local_rank == 0 and epoch % 1 == 0:  # only one GPU is used to store the model
            os.makedirs(args.loss_path, exist_ok=True)
            print(f"loss save to {args.loss_path}")
            np.save(args.loss_path + "/cl_loss_" + str(epoch+1) + "_epoch.npy",
                    torch.tensor(epoch_losses).cpu().numpy())
            os.makedirs(args.model_path, exist_ok=True)
            print(f"model save to {args.model_path}")
            torch.save(model.module.state_dict(
            ), args.model_path + "/TSFE_" + str(epoch+1) + "_epoch.pkl")

    if args.nprocs > 1:  # if use distribution training
        torch.distributed.destroy_process_group()  # close process


def parse_args():
    parser = argparse.ArgumentParser(description="TSFE")

    parser.add_argument("--workers", type=int, default=4,
                        help="number of workers in DataLoader")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="batch size per GPU")  # batch size default is the total number of batchsize
    parser.add_argument("--img_size", type=tuple,
                        default=(256, 256), help="image size ")
    parser.add_argument('--lr', type=float, default=0.6,
                        help='learning rate')

    parser.add_argument('--encoder_network', type=str, default='resnet50',
                        help='encoder network architecture')
    parser.add_argument('--epochs', type=int, default=200,
                        help='total number of epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--patch_dim', type=tuple, default=(2, 2),
                        help='number of patches for each input image')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='dimensions of the output feature from projector')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='temperature parameter')
    parser.add_argument("--model_path", type=str,
                        default="./results/models", help="folder to save trained models")  # path to store model
    parser.add_argument("--loss_path", type=str,
                        default="./results/loss", help="folder to save trained loss")  # path to store loss
    parser.add_argument("--distorted_image_path_pre", type=str, default="./exploration_database_and_code/distort_images",
                        help="path for distorted training image")  # path to store distorted image
    parser.add_argument("--pristine_image_path_pre", type=str, default="./exploration_database_and_code/pristine_images",
                        help="path for reference training image")  # path to store original image

    # training parameters for distribution training
    parser.add_argument("--nprocs", type=int, default=4,
                        help="number of process")  # one process is resposible for one GPU, so it is also equal to the number of GPUs

    args = parser.parse_args()
    args.ngpus_per_proc = 1  # the number of gpu for each process
    args.world_size = args.ngpus_per_proc * args.nprocs   # number of global process

    args.device = torch.device(
        "cuda:4" if torch.cuda.is_available() else "cpu")

    args.num_patches = args.patch_dim[0]*args.patch_dim[1]
    args.n_features = None
    return args


if __name__ == "__main__":
    print("---------start-----------")
    # parameters in this epoch of training is stored in args
    args = parse_args()

    if torch.cuda.is_available():
        print("Cuda is available!")
        if torch.cuda.device_count() >= args.nprocs:
            print(
                f"Training with {args.nprocs} GPUs on {args.encoder_network}.")
            mp.spawn(fn=main, nprocs=args.nprocs, args=(args,))
        else:
            main(0, args)
    print("---------end-----------")
