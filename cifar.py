'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.cifar import Avd_NIN, resnet

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = ['NIN']

class NINWrapper(nn.Module):
    def __init__(self, _num_stages=3, _use_avg_on_conv3=True, indim=192, # 384, 
        num_classes=10):
        super(NINWrapper, self).__init__()
        self.nin = Avd_NIN(_num_stages=_num_stages, _use_avg_on_conv3=_use_avg_on_conv3)
        self.fc = nn.Linear(indim, num_classes)
    
    def forward(self, x, im_type):
        x = self.nin(x, im_type, out_feat_keys=None)
        return self.fc(x)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.256, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--step-size', type=int, default=3,
                    help='Decrease learning rate each step-size epochs.')
parser.add_argument('--gamma', type=float, default=0.97, 
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Attack options
parser.add_argument('--epsilon', type=float, default=1.0, 
    help='attack strength')
parser.add_argument('--advprop-lambda', type=float, default=0.0, 
    help='multiplier for advprop loss term')


args = parser.parse_args()
args.step_size = 1. / 255. # alpha
# paper's heuristic for num of attack iters
args.attack_iters = 1 if int(args.epsilon) == 1 else int(args.epsilon + 1)
args.epsilon = args.epsilon / 255.
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset in ['cifar10', 'cifar100'], 'Dataset should be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
device = torch.device('cuda' if use_cuda else 'cpu')
def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    # Data
    print('==> Preparing dataset %s' % args.dataset)
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100
    
    mean, std = mean.to(device), std.to(device)
    args.actual_epsilon = args.epsilon / std
    args.actual_epsilon = args.actual_epsilon.view(1, 3, 1, 1)
    if args.advprop_lambda > 0.0:
        print(f'Running AdvProp with lambda = {args.advprop_lambda}, '
            f'epsilon = {args.epsilon}, ({args.actual_epsilon.squeeze()}), '
            f'n = {args.attack_iters}, and alpha = {args.step_size}')
    
    lower_limit, upper_limit = (0.0 - mean) / std, (1.0 - mean) / std
    lower_limit = lower_limit.view(1,3,1,1)
    upper_limit = upper_limit.view(1,3,1,1)

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    model = resnet(depth=20, num_classes=num_classes)
    # model = Avd_NIN(num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0.9, 
    #     momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.9) # ,
        # momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, 
        gamma=args.gamma)

    # Resume
    title = f'{args.dataset}-{args.arch}'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint...')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # Set BN's momentum and weight decay
    tot_bns = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            tot_bns += 1
            # module.momentum = 0.99
    print(f'Found and modified the momentum of {tot_bns} BatchNorm layers')

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, 
            epoch, lower_limit, upper_limit, args)
        test_loss, test_acc = test(testloader, model, criterion, epoch)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)
        scheduler.step()

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, lower_limit, 
        upper_limit, args):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        # compute output
        outputs = model(inputs) # , im_type='nat')
        loss = criterion(outputs, targets)
        # compute adv examples
        if args.advprop_lambda > 0.0:
            adv_inputs = attack_pgd(
                model, inputs, targets, epsilon=args.actual_epsilon, 
                alpha=args.step_size, attack_iters=args.attack_iters, restarts=1, 
                lower_limit=lower_limit, upper_limit=upper_limit
            )
            # adv outputs
            adv_outputs = model(adv_inputs, im_type='adv')
            adv_loss = criterion(adv_outputs, targets)
            loss = loss + adv_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        # compute output
        outputs = model(inputs) # , im_type='nat')
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


# ======================================================
# Utils for computing adversarial examples
# Taken from
# https://github.com/anonymous-sushi-armadillo/fast_is_better_than_free_CIFAR10/blob/master/evaluate_cifar.py#L45
def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit,
        upper_limit):
    def clamp(X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)
    
    max_loss = torch.zeros(y.shape[0], device=X.device)
    max_delta = torch.zeros_like(X)
    for _ in range(restarts):
        delta = torch.zeros_like(X)
        delta[:, 0, :, :].uniform_(-epsilon[0, 0].item(), epsilon[0, 0].item())
        delta[:, 1, :, :].uniform_(-epsilon[0, 1].item(), epsilon[0, 1].item())
        delta[:, 2, :, :].uniform_(-epsilon[0, 2].item(), epsilon[0, 2].item())
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta, im_type='adv') # Using adversarial BNs
            index = torch.nonzero(output.max(1)[1] == y) # torch.where(output.max(1)[1] == y)
            if len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta, im_type='adv'), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


if __name__ == '__main__':
    main()
