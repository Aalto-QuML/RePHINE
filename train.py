import torch


def train(loader, model, loss_fn, optimizer, device):
    model.train()
    train_losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, evaluator=None):
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = loss_fn(out.squeeze(), batch.y)
        acc = -loss
        if not isinstance(loss_fn, torch.nn.L1Loss):
            acc = (out.argmax(dim=-1) == batch.y.squeeze()).float().mean()

    if evaluator is not None:
        acc = evaluator.eval({'y_pred': out[:, 1].unsqueeze(dim=1), 'y_true': batch.y})[evaluator.eval_metric]

    return loss, acc


def train_perslay(features, diags, labels, model, loss_fn, optimizer, device, batch_size=128):
    model.train()
    train_losses = []
    train_num_pts = features.shape[0]
    for batch_start in range(0, train_num_pts, batch_size):
        batch_end = min(batch_start + batch_size, train_num_pts)
        batch_label = labels[batch_start:batch_end].to(device)
        batch_feats = features[batch_start:batch_end].to(device)
        batch_diags = [diag[batch_start:batch_end].to(device) for diag in diags]

        optimizer.zero_grad()
        outputs = model([batch_diags, batch_feats])
        loss = loss_fn(outputs, batch_label)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    return train_losses

@torch.no_grad()
def evaluate_perslay(model, features, diags, labels, loss_fn, device):
    model.eval()
    batch_label = labels.to(device)
    batch_feats = features.to(device)
    batch_diags = [diag.to(device) for diag in diags]

    out = model([batch_diags, batch_feats])
    loss = loss_fn(out, batch_label)
    acc = (out.argmax(dim=-1) == batch_label.argmax(dim=-1)).float().mean()

    return loss, acc
