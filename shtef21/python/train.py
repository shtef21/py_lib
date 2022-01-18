
import torch
from torch import nn
from torch.utils.data import DataLoader as DL

from tqdm import tqdm 

def train(model, **kwargs):

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
    trainset, testset = arg_req('trainset')

    # Check params
    assert not any(missing_kwargs), f'Missing required args: {missing_kwargs}'

    train_loader = DL(trainset, batch_size, pin_memory=True, shuffle=True)
    test_loader = DL(testset, batch_size_test, pin_memory=True)

    for e_curr in range(epoch, epochs):

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
            correct += corr
            total += tot

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

        # End of epoch

    # End of "train"
