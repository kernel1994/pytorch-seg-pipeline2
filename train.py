import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import CrossEntropyLoss

import cfg
import utils


def _train_epoch(model, epoch, dataloader, optimizer):
    model.train()  # set model to training mode

    for batch_idx, (images, masks, _) in enumerate(dataloader, start=1):
        images = images.to(cfg.device)
        masks = masks.long().to(cfg.device)

        optimizer.zero_grad()
        output = model(images)
        loss = CrossEntropyLoss()(output, masks)
        loss.backward()
        optimizer.step()

        print(f'epoch [{epoch}/{cfg.max_epoch}]', end=' | ')
        print(f'train iter [{batch_idx}/{len(dataloader)}]', end=' | ')
        print(f'loss={loss.item():.5f}')


def _valid_epoch(model, epoch, dataloader, best_loss, results_dir):
    model.eval()  # set model to evaluation mode

    total_val_loss = 0.
    best_val_loss = best_loss
    with torch.no_grad():
        for batch_idx, (images, masks, _) in enumerate(dataloader, start=1):
            images = images.to(cfg.device)
            masks = masks.long().to(cfg.device)

            output = model(images)
            loss = CrossEntropyLoss()(output, masks)

            total_val_loss += loss.item()

            print(f'epoch [{epoch}/{cfg.max_epoch}]', end=' | ')
            print(f'val iter [{batch_idx}/{len(dataloader)}]', end=' | ')
            print(f'val loss={loss.item():.5f}')

        mean_val_loss = total_val_loss / len(dataloader)
        if mean_val_loss < best_loss:
            torch.save(model.state_dict(), results_dir.joinpath(cfg.best_model_path_name))

            best_val_loss = mean_val_loss

    print(f'epoch [{epoch}/{cfg.max_epoch}]', end=' | ')
    print(f'val', end=' | ')
    print(f'mean loss={mean_val_loss:.5f}', end=' | ')
    print(f'best loss={best_val_loss:.5f}')

    return best_val_loss


def train_val(model, train_dataloader, test_dataloader, results_dir):
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)

    best_loss = float('inf')
    for epoch in range(cfg.max_epoch):
        _train_epoch(model, epoch, train_dataloader, optimizer)
        best_loss = _valid_epoch(model, epoch, test_dataloader, best_loss, results_dir)
        schedular.step()


def test_batch(model, dataloader, results_dir):
    model_path = results_dir.joinpath(cfg.best_model_path_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for images, masks, p_names in dataloader:
            images = images.to(cfg.device)
            masks = masks.long().to(cfg.device)

            output = model(images)

            positive_prob = F.softmax(output, dim=1)[:, 1]
            for iter_idx in range(output.size(0)):
                prd_mask = positive_prob[iter_idx].detach().cpu().numpy().reshape((cfg.w, cfg.h))

                p_name = p_names[iter_idx]
                print(f'now testing {p_name}')

                np.save(results_dir.joinpath(f'{p_name}_prd_prob.npy'), prd_mask)
                utils.postprocess(prd_mask, results_dir.joinpath(f'{p_name}_prd_bin.png'))
