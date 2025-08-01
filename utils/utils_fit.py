import random
import torch
from tqdm import tqdm
import torch.nn as nn
from utils.utils import get_lr
from torchvision.utils import save_image

def augment(inp_img):
    res = []
    for img in inp_img:
        aug = random.randint(0, 8)
        if aug == 1:
            img = img.flip(1)
        elif aug == 2:
            img = img.flip(2)
        elif aug == 3:
            img = torch.rot90(img, dims=(1, 2))
        elif aug == 4:
            img = torch.rot90(img, dims=(1, 2), k=2)
        elif aug == 5:
            img = torch.rot90(img, dims=(1, 2), k=3)
        elif aug == 6:
            img = torch.rot90(img.flip(1), dims=(1, 2))
        elif aug == 7:
            img = torch.rot90(img.flip(2), dims=(1, 2))
        res.append(img)
    return torch.stack(res, dim=0)

def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, save_period):
    import os
    os.makedirs("logs", exist_ok=True)
    device = torch.device("cuda")
    Det_loss = 0
    val_loss = 0

    criterion_l1 = nn.L1Loss().to(device)
    contrast_loss = nn.CrossEntropyLoss().to(device)
    wgt = [1.0, 0.9, 0.8, 0.7, 0.6]

    model_train.train()
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets, clearimgs = batch[0], batch[1], batch[2]
            with torch.no_grad():
                images = torch.from_numpy(images).float().to(device)
                targets = [torch.from_numpy(ann).float().to(device) for ann in targets]
                clearimgs = torch.from_numpy(clearimgs).float().to(device)
                posimgs = augment(images)
                hazy_and_clear = torch.cat([images, posimgs], dim=0).to(device)

            optimizer.zero_grad()

            detected, restored, logits, labels = model_train(hazy_and_clear)

            loss_det = yolo_loss(detected, targets)
            restored = restored[:images.shape[0]]
            loss_l1 = criterion_l1(restored, clearimgs)
            loss_contrs = contrast_loss(logits, labels)
            total_loss = 0.2 * loss_det + wgt[epoch // 20] * loss_l1 + 0.1 * loss_contrs

            total_loss.backward()
            optimizer.step()

            Det_loss += loss_det.item()

            pbar.set_postfix(**{
                'loss_det': f'{loss_det:.2f}',
                'loss_l1': f'{loss_l1:.2f}',
                'loss_contrs': f'{loss_contrs:.2f}',
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                images = torch.from_numpy(images).float().to(device)
                # Nếu ảnh đầu vào là haze và clear ghép thành tuple: (haze, clear)
                if isinstance(images, (list, tuple)) and len(images) == 2:
                    images = torch.cat([images[0], images[1]], dim=1)  # ghép theo chiều kênh (B, 6, H, W)

                targets = [torch.from_numpy(ann).float().to(device) for ann in targets]

                outputs = model_train(images)
                if isinstance(outputs, (list, tuple)):
                    if len(outputs) == 4:
                        detected, _, _, _ = outputs
                    elif len(outputs) == 2:
                        detected, _ = outputs
                    else:
                        detected = outputs[0]
                else:
                    detected = outputs


                det_loss = yolo_loss(detected, targets)


            val_loss += det_loss.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    loss_history.append_loss(epoch + 1, Det_loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (Det_loss / epoch_step, val_loss / epoch_step_val))

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (
            epoch + 1, Det_loss / epoch_step, val_loss / epoch_step_val))
