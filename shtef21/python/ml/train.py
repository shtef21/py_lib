
import torch
from torch import nn
from torch.utils.data import DataLoader as DL

from tqdm import tqdm 

def train(model, **kwargs):
    """
    Pytorch train wrapper.

    Args:
        model - a Neural network model

    Kwargs:
        (required) epochs -- total epochs \n
        (required) criterion -- loss function \n
        (required) optimizer -- model optimizer \n
        (required) trainset -- train dataset \n
        testset -- if sent, tests the model post epoch \n
        device -- force a device (default: cuda if exist else cpu) \n
        batch_size -- train loader batch size (default: 256) \n
        lr_scheduler -- optional post-epoch scheduler \n
        batch_f -- runs post batch, receives batch stats \n
        epoch_f -- runs post epoch, receives epoch stats \n
        test_once -- if True, only tests model after last epoch

    """

    missing_kwargs = []

    # Required kwarg
    def arg_req(p, default=None):
        if p in kwargs:
            return kwargs[p]
        missing_kwargs.append(p)
        return default

    # Optional kwarg
    def arg_opt(p, default=None):
        return kwargs[p] if p in kwargs else default 
    
    # Post batch and post epoch function
    batch_f = arg_opt('batch_f')
    epoch_f = arg_opt('epoch_f')

    device = arg_opt('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = arg_opt('batch_size', default=256)
    batch_size_test = 16
    epoch, epochs = arg_opt('epoch', 0), arg_req('epochs')

    model = model.to(device)
    criterion = arg_req('criterion')
    optimizer = arg_req('optimizer')
    lr_scheduler = arg_opt('lr_scheduler')
    trainset = arg_req('trainset')
    testset = arg_opt('testset')
    has_test = testset != None
    test_once = arg_opt('test_once', False)

    # Check params
    assert not any(missing_kwargs), f'Missing required args: {missing_kwargs}'

    train_loader = DL(trainset, batch_size, pin_memory=True, shuffle=True)
    test_loader = DL(testset, batch_size_test, pin_memory=True) if has_test else None

    for e_curr in range(epoch, epochs):

        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        loop.set_description(f'e={e_curr + 1}/{epochs}')
        correct, total = 0, 0

        for idx, (images, labels) in loop:

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(out, -1)
            corr, tot = (preds == labels).sum().item(), len(labels)
            acc = corr / tot
            correct += corr
            total += tot

            loop.set_postfix(acc=round(corr / tot, 4))

            if batch_f != None:
                batch_f({
                    'loss': loss.item(),
                    'batch_result': {
                        'correct': corr,
                        'total': tot,
                        'acc': corr / tot,
                    },
                })

            # End of batch

        if lr_scheduler != None:
            lr_scheduler.step()

        if epoch_f != None:
            epoch_f({
                'epoch_result': {
                    'correct': correct,
                    'total': total,
                    'acc': correct / total,
                },
            })

        if has_test and (test_once == False or e_curr + 1 == epochs):
            with torch.no_grad():

                model.eval()
                test_loop = tqdm(enumerate(test_loader), total=len(test_loader))
                test_loop.set_description(f'Test e={e_curr + 1}')
                correct, total = 0, 0

                for idx, (images, labels) in test_loop:

                    images, labels = images.to(device), labels.to(device)
                    out = model(images)
                    preds = torch.argmax(out, -1)
                    correct += (preds == labels).sum().item()
                    total += len(labels)
                    
                    test_loop.set_postfix(acc=round(correct / total, 4))

        # End of epoch

    # End of "train"
