"""A shared recurrent encoder; task heads and masked-event pretraining."""

import copy

import numpy as np
import torch
from torch import nn

from .data import SEQUENCE_LENGTH

HIDDEN = 32
BATCH_SIZE = 256
LEARNING_RATE = 0.003
MASK_RATE = 0.35


class EventEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 8, padding_idx=0)
        self.recurrent = nn.GRU(12, HIDDEN, batch_first=True)

    def forward(self, tokens, values):
        hidden, _ = self.recurrent(torch.cat((self.embedding(tokens), values), dim=-1))
        return hidden


class PassModel(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else EventEncoder()
        self.heads = nn.Linear(HIDDEN, 2)

    def forward(self, tokens, values):
        return self.heads(self.encoder(tokens, values)[:, -1])


class ContextMLP(nn.Module):
    """The same observed history, flattened with fixed event-type one-hot fields."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(SEQUENCE_LENGTH * 15, HIDDEN), nn.ReLU())
        self.heads = nn.Linear(HIDDEN, 2)

    def forward(self, tokens, values):
        event_types = nn.functional.one_hot(tokens, num_classes=11).to(values.dtype)
        context = torch.cat((values, event_types), dim=-1).flatten(1)
        return self.heads(self.encoder(context))


def tensors(data):
    return tuple(torch.from_numpy(v) for v in data)


def task_loss(prediction, labels, task="both"):
    completion = nn.functional.binary_cross_entropy_with_logits(
        prediction[:, 0], labels[:, 0]
    )
    progression = nn.functional.mse_loss(prediction[:, 1], labels[:, 1])
    return (
        completion
        if task == "completion"
        else progression if task == "progression" else completion + progression
    )


def fit_supervised(
    model, data, *, epochs, seed, validation=None, task="both", freeze=False
):
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not freeze)
    tokens, values, labels = tensors(data)
    valid = tensors(validation) if validation is not None else None
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    generator = torch.Generator().manual_seed(seed)
    best, state, history = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        total = 0
        for indices in torch.randperm(len(tokens), generator=generator).split(
            BATCH_SIZE
        ):
            optimizer.zero_grad()
            loss = task_loss(
                model(tokens[indices], values[indices]), labels[indices], task
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            score = (
                float(task_loss(model(valid[0], valid[1]), valid[2], task))
                if valid
                else total / len(tokens)
            )
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": total / len(tokens),
                "validation_loss": score if valid else None,
            }
        )
        if score < best:
            best, state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return history


def pretrain_masked(data, *, epochs, seed):
    """Mask observed event types/starts; never access completion/displacement labels."""
    torch.manual_seed(seed)
    encoder = EventEncoder()
    decoder = nn.Linear(HIDDEN, 12)  # two coordinates, ten event types
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE
    )
    tokens, values = tensors(data[:2])
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        total, count = 0, 0
        for indices in torch.randperm(len(tokens), generator=generator).split(
            BATCH_SIZE
        ):
            t, v = tokens[indices], values[indices]
            mask = (torch.rand(t.shape, generator=generator) < MASK_RATE) & (t > 0)
            if not mask.any():
                continue
            masked_t, masked_v = t.clone(), v.clone()
            masked_t[mask] = 11
            masked_v[..., :2][mask] = 0
            optimizer.zero_grad()
            prediction = decoder(encoder(masked_t, masked_v))[mask]
            loss = nn.functional.mse_loss(
                prediction[:, :2], v[..., :2][mask]
            ) + nn.functional.cross_entropy(prediction[:, 2:], t[mask] - 1)
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * int(mask.sum())
            count += int(mask.sum())
        history.append(
            {"epoch": epoch + 1, "masked_reconstruction_loss": total / count}
        )
    return encoder, history


def predict(model, data):
    model.eval()
    tokens, values = tensors(data[:2])
    with torch.no_grad():
        outputs = torch.cat(
            [model(t, v) for t, v in zip(tokens.split(2048), values.split(2048))]
        )
    return np.column_stack((outputs[:, 0].sigmoid().numpy(), outputs[:, 1].numpy()))


def load_model(path):
    """Load the experiment's numeric NPZ state, with pickle explicitly disabled."""
    model = PassModel()
    with np.load(path, allow_pickle=False) as arrays:
        model.load_state_dict(
            {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files},
            strict=True,
        )
    model.eval()
    return model
