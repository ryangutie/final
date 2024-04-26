# https://docs.python.org/3/reference/index.html
from .model import PuckDetector, save_model, load_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
from torchvision import transforms as T

def train(args):
    from os import path
    import timeit
    from tqdm import tqdm

    # Initialize timer for training duration tracking
    start = timeit.default_timer()

    # Set device for training based on CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device = ', device)
    print(torch.cuda.get_device_name(device))

    # Setup tensorboard logging if directory is provided
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'fcmtrain' + args.run), flush_secs=1)
        params_logger = tb.SummaryWriter(path.join(args.log_dir, 'fcmparams' + args.run), flush_secs=1)

    # Log learning rate, epoch, and kernel size
    params_logger.add_text('lrate', str(args.lrate))
    params_logger.add_text('epoch', str(args.epoch))
    params_logger.add_text('kernel', str(args.kernel))
    params_logger.add_text('loss', str(args.loss))

    # Setup model with specified layers and move to designated device
    layers = [16, 32, 64, 128]
    params_logger.add_text('layers', str(layers))
    model = load_model().to(device)

    # Setup optimizer and loss function based on user selection
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=1e-5)
    if args.loss == "mse":
        print("mse loss is going to be used")
        loss = torch.nn.MSELoss()
    elif args.loss == "l1":
        print("l1 loss being used")
        loss = torch.nn.L1Loss()

    # Log and print data loading phase
    print('======================= loading data =======================')
    transform = dense_transforms.Compose([
        dense_transforms.ColorJitter(0.4, 0.8, 0.7, 0.3),
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor()
    ])
    train_data = load_data(args.trainPath, num_workers=4, batch_size=64, transform=transform)

    # Execute training across specified number of epochs
    global_step = 0
    print("Setup Complete, starting to train on {epoch} epochs!".format(epoch=args.epoch))
    for epoch in range(args.epoch):
        print('======================= training epoch', epoch, '=======================')

        for img, label in tqdm(train_data):
            model.train()
            img, label = img.to(device), label.to(device)

            # Perform forward pass and calculate loss
            prediction = model(img)
            loss_val = loss(prediction, label).to(device)

            # Log loss every 100 steps
            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        # Evaluate and save model after each epoch
        model.eval()
        save_model(model)

    # Final evaluation and model saving
    model.eval()
    save_model(model)
    train_time = timeit.default_timer()
    print("Done!! Took {}, start {}, {}".format(train_time - start, start, train_time))

if __name__ == '__main__':
    import argparse

    # Setup command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default=r'tmpLogs')
    parser.add_argument('--run', default='46')
    parser.add_argument('-e', '--epoch', default=20)
    parser.add_argument('-t', '--trainPath', default=r'data/train')
    parser.add_argument('-l', '--lrate', default=0.001)
    parser.add_argument('--kernel', default=3)
    parser.add_argument('--loss', default='mse')

    args = parser.parse_args()
    train(args)
