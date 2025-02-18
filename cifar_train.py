# original code: https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py
import random
import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from tensorboardX import SummaryWriter
from imbalance_data.imbalance_cifar import IMBALANCECIFAR100
from imbalance_data.dnet_loader import BaseLoader
from losses import LDAMLoss, BalancedSoftmaxLoss
from opts import parser
import warnings
from util.util import *
from util.autoaug import CIFAR10Policy, Cutout
import util.moco_loader as moco_loader


best_acc1 = 0
is_best = 0

def main():
    args = parser.parse_args()
    # args.store_name = '_'.join(
    #     [args.dataset, args.arch, args.loss_type, args.train_rule, args.data_aug, str(args.imb_factor_src),
    #      str(args.rand_number),
    #      str(args.mixup_prob), args.exp_str])
    # import pdb; pdb.set_trace()
    prepare_folders(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda


    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = args.num_classes
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.Backbone(args.arch, use_norm, args.bn_dim, args.n_classes)
    # print(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    params = [{"params":model.model_fc.parameters() , "lr_mult":1, "decay_mult":2}] + \
             [{"params":model.bottleneck_layer.parameters(), "lr_mult":0.1, "decay_mult":2}] + \
             [{"params":model.classifier_layer.parameters(), "lr_mult":0.1, "decay_mult":2}] 

    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # scheduler = models.inv_scheduler(optimizer, gamma=args.gamma, power=args.power)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['iter']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            # scheduler.load_state_dict(checkpoint['scheduler_dict'])
            print("=> loaded checkpoint '{}' (iter {})"
                  .format(args.resume, checkpoint['iter']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    if args.use_randaug:
        """
        if use_randaug == True, we follow randaug following PaCo's setting (ICCV'2021),
        400 epoch & Randaug 
        https://github.com/dvlab-research/Parametric-Contrastive-Learning/blob/main/LT/paco_cifar.py
        """
        print("use randaug!!")
        augmentation_regular = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),  # add AutoAug
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]

        augmentation_sim_cifar = [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation_sim_cifar)]

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    print(args)

    # train_dataset = IMBALANCECIFAR100(root=args.root, imb_factor=args.imb_factor,
    #                                   rand_number=args.rand_number, weighted_alpha=args.weighted_alpha, train=True, download=True,
    #                                   transform=transform_train, use_randaug=args.use_randaug)
    # val_dataset = datasets.CIFAR100(root=args.root, train=False, download=True, transform=transform_val)

    source, target = args.source, args.target

    ## source and target - train
    train_dataset_source = BaseLoader(root=args.root, txt=source, transform=transform_train, use_randaug=args.use_randaug, weighted_alpha=args.weighted_alpha, imb_factor=args.imb_factor_src, cls_num=args.n_classes)

    train_dataset_target = BaseLoader(root=args.root, txt=target, transform=transform_train, use_randaug=args.use_randaug, weighted_alpha=args.weighted_alpha, imb_factor=args.imb_factor_tgt, cls_num=args.n_classes)

    if args.dataset == "DomainNet":
        source = source.replace("train" , "test")
        target = source.replace("train" , "test")
    
    ## source and target - val
    val_dataset_source = BaseLoader(root=args.root, txt=source, transform=transform_val, cls_num=args.n_classes)

    val_dataset_target = BaseLoader(root=args.root, txt=target, transform=transform_val, cls_num=args.n_classes)


    cls_num_list = train_dataset_source.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    train_cls_num_list = np.array(cls_num_list)

    if args.CBS:
        train_sampler = BalancedSampler(train_dataset_source)
    else:
        train_sampler = None

    train_loader_source = torch.utils.data.DataLoader(
            train_dataset_source, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader_source = torch.utils.data.DataLoader(
        val_dataset_source, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    weighted_train_loader = None
    weighted_cls_num_list = [0] * num_classes


    train_loader_target = torch.utils.data.DataLoader(
            train_dataset_target, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader_target = torch.utils.data.DataLoader(
        val_dataset_target, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.data_aug == 'CMO':
        weighted_sampler = train_dataset_source.get_weighted_sampler()
        weighted_train_loader = torch.utils.data.DataLoader(
            train_dataset_source, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, sampler=weighted_sampler)
        inverse_iter = loop_iterable(weighted_train_loader)

    batch_iterator = zip(loop_iterable(train_loader_source), loop_iterable(train_loader_target))
    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    start_time = time.time()
    print("Training started!")

    is_best = 0
    for iter in range(args.start_epoch, args.epochs):

        if args.use_randaug:
            paco_adjust_learning_rate(optimizer, iter, args)
        else:
            adjust_learning_rate(optimizer, iter, args)

        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'CBReweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = iter // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')


        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'BS':
            criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list_cuda).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one iter
        # switch to train mode
        model.train()
        train(batch_iterator, model, criterion, optimizer, iter, args, log_training,
              tf_writer, inverse_iter)
        # scheduler.step()

        # evaluate on validation set
        if (iter + 1) % args.valid_freq == 0:
            acc1_source = validate(val_loader_source, model, criterion, iter, args, log_testing, tf_writer)

            # acc1_target = validate(val_loader_target, model, criterion, iter, args, log_testing, tf_writer)

            # remember best acc@1 and save checkpoint
            is_best = acc1_source > best_acc1
            best_acc1 = max(acc1_source, best_acc1)

            tf_writer.add_scalar('acc/test_top1_best', best_acc1, iter)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            log_testing.write(output_best + '\n')
            log_testing.flush()

        save_checkpoint(args, {
            'iter': iter + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            # 'scheduler_dict': scheduler.state_dict()
        }, is_best, iter + 1)

    end_time = time.time()

    print("It took {} to execute the program".format(hms_string(end_time - start_time)))
    log_testing.write("It took {} to execute the program".format(hms_string(end_time - start_time)) + '\n')
    log_testing.flush()


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def train(train_loader, model, criterion, optimizer, iter, args, log,
              tf_writer, inverse_iter=None):
    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')


    end = time.time()

    input_src, target_src, input_tgt, _ = next(train_loader)
    if args.data_aug == 'CMO' and args.start_data_aug < iter < (args.epochs - args.end_data_aug):
        input2, target2 = next(inverse_iter)
        input2 = input2[:input_src.size()[0]]
        target2 = target2[:target_src.size()[0]]
        input2 = input2.cuda(args.gpu, non_blocking=True)
        target2 = target2.cuda(args.gpu, non_blocking=True)

    # measure data loading time
    # data_time.update(time.time() - end)

    input_src = input_src.cuda(args.gpu, non_blocking=True)
    target_src = target_src.cuda(args.gpu, non_blocking=True)
    input_tgt = input_tgt.cuda(args.gpu, non_blocking=True)
    # Data augmentation
    r = np.random.rand(1)

    if args.data_aug == 'CMO' and args.start_data_aug < iter < (args.epochs - args.end_data_aug) and r < args.mixup_prob:
        # generate mixed sample
        lam = np.random.beta(1, 1)
        bbx1, bby1, bbx2, bby2 = rand_bbox(input_src.size(), lam)
        input_src[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input_src.size()[-1] * input_src.size()[-2]))
        # compute output
        output = model(input_src)
        loss = criterion(output, target_src) * lam + criterion(output, target2) * (1. - lam)

    else:
        output = model(input_src)
        loss = criterion(output, target_src)

    # measure accuracy and record loss
    # acc1, acc5 = accuracy(output, target_src, topk=(1, 5))
    # losses.update(loss.item(), input_src.size(0))
    # top1.update(acc1[0], input_src.size(0))
    # top5.update(acc5[0], input_src.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    # batch_time.update(time.time() - end)
    # end = time.time()

    if iter % args.print_freq == 0:
        output = ('iter: [{0}/{1}], lr: {lr:.5f}\t'.format(
            iter, args.epochs, lr=optimizer.param_groups[-1]['lr']))  # TODO
        print(output)
        log.write(output + '\n')
        log.flush()

    tf_writer.add_scalar('loss/train', loss.item(), iter)
    # tf_writer.add_scalar('acc/train_top1', top1.avg, iter)
    # tf_writer.add_scalar('acc/train_top5', top5.avg, iter)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], iter)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def rand_bbox_withcenter(size, lam, cx, cy):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(val_loader, model, criterion, iter, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        # print(out_cls_acc)

        if args.imb_factor == 0.01:
            many_shot = train_cls_num_list > 100
            medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
            few_shot = train_cls_num_list < 20
            print("many avg, med avg, few avg", float(sum(cls_acc[many_shot]) * 100 / sum(many_shot)),
                  float(sum(cls_acc[medium_shot]) * 100 / sum(medium_shot)),
                  float(sum(cls_acc[few_shot]) * 100 / sum(few_shot)))

        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_' + flag, losses.avg, iter)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, iter)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, iter)
        tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, iter)

    return top1.avg


def adjust_learning_rate(optimizer, iter, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    iter = iter + 1
    if iter <= 100:
        lr = args.lr #* iter / 100
    elif iter > args.epochs*.9:
        lr = args.lr * 0.0001
    elif iter > args.epochs*.8:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def paco_adjust_learning_rate(optimizer, iter, args):
    # experiments as PaCo (ICCV'21) setting.
    warmup_epochs = 1000
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if iter <= warmup_epochs:
        lr = args.lr #/ warmup_epochs * (iter + 1)
    elif iter > args.epochs*.9:
        lr = args.lr * 0.01
    elif iter > args.epochs*.8:
        lr = args.lr * 0.1
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
