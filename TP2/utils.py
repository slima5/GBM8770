import random
from typing import overload

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as T
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class _ToRGB(T.Transform):
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 3 or t.size(0) != 1:
            raise ValueError(f"Expected a 3D tensor with 1 channel, got {t.shape}")
        return t.expand(3, -1, -1)


class _Transform:
    def __init__(self, data_aug: bool = False):
        transform_list = []
        if data_aug:
            transform_list.extend([T.RandomResizedCrop((224, 224)), T.RandomHorizontalFlip()])
        else:
            transform_list.extend([T.Resize((256, 256)), T.CenterCrop((224, 224))])
        transform_list.extend([T.ToImage(), T.ToDtype(torch.float32, scale=True), _ToRGB()])
        self._transform = T.Compose(transform_list)

    def __call__(self, data: dict[str, list[torch.Tensor]]) -> dict[str, list[torch.Tensor]]:
        data["image"] = [self._transform(image) for image in data["image"]]
        return data


def get_dataloaders(
    ds: datasets.DatasetDict,
    *,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_subset: float | None = None,
    data_aug: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Crée les dataloaders pour l'entraînement, la validation et le test.

    Args:
        ds (datasets.DatasetDict): le dataset contenant les splits "train" et "test".
        batch_size (int): la taille des batches.
        num_workers (int, optional): le nombre de workers pour le DataLoader. Par défaut à 4.
        pin_memory (bool, optional): si True, utilise le pin_memory pour le DataLoader. Par défaut à True.
        train_subset (float, optional): si spécifié, utilise seulement cette fraction des données d'entraînement. Doit
            être dans (0, 1]. Par défaut à None, ce qui utilise toutes les données d'entraînement.
        data_aug (bool, optional): si True, utilise des augmentations de données pour l'entraînement. Par défaut à
            False.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: les dataloaders pour l'entraînement, la validation et le test.
    """
    # On utilise 5% des données d'entraînement pour la validation
    ds = ds.copy()
    _ds = ds["train"].train_test_split(test_size=0.05, seed=42)
    ds["train"] = _ds["train"]
    ds["validation"] = _ds["test"]

    if train_subset is not None:
        assert 0 < train_subset <= 1, "train_subset must be in (0, 1]"
        ds["train"] = ds["train"].shuffle(seed=42).select(list(range(int(len(ds["train"]) * train_subset))))

    transform = _Transform(data_aug=data_aug)

    train_loader = DataLoader(
        ds["train"].with_transform(transform),
        shuffle=True,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        ds["validation"].with_transform(transform),
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        ds["test"].with_transform(transform),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
) -> list[float]:
    """Entraîne le modèle sur une époque.

    Args:
        model (nn.Module): le modèle à entraîner.
        loader (DataLoader): le dataloader pour l'entraînement.
        optimizer (Optimizer): l'optimiseur à utiliser.

    Returns:
        list[float]: la liste des valeurs de loss pour chaque batch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    losses = []
    for batch in tqdm(loader, leave=False, desc="Training"):
        images, targets = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        assert logits.shape == (images.size(0), 4), f"Expected logits shape {(images.size(0), 4)}, got {logits.shape}"
        assert targets.shape == (images.size(0),), f"Expected targets shape {(images.size(0),)}, got {targets.shape}"
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


@overload
def validate(
    model: nn.Module,
    loader: DataLoader,
    *,
    return_preds: bool = False,
) -> tuple[float, float]: ...
@overload
def validate(
    model: nn.Module,
    loader: DataLoader,
    *,
    return_preds: bool = True,
) -> tuple[float, float, list[int]]: ...
def validate(
    model: nn.Module,
    loader: DataLoader,
    *,
    return_preds: bool = False,
) -> tuple[float, float, list[int]] | tuple[float, float]:
    """Évalue le modèle sur l'ensemble de validation.

    Args:
        model (nn.Module): le modèle à évaluer.
        loader (DataLoader): le dataloader pour la validation.
        return_preds (bool, optional): Si True, retourne aussi les prédictions du modèle. Par défaut, False. À noter que
            ce paramètre doit être spécifié par nom si on souhaite utiliser une valeur autre que la valeur par défaut.

    Returns:
        tuple[float, float] ou tuple[float, float, list[int]]: la loss moyenne, l'accuracy, et éventuellement les
        prédictions du modèle.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    sum_loss = 0
    num_correct = 0
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="Validation"):
            images, targets = batch["image"].to(device), batch["label"].to(device)
            logits = model(images)
            sum_loss += F.cross_entropy(logits, targets, reduction="sum").item()
            preds_ = logits.argmax(dim=1)
            num_correct += preds_.eq(targets.view_as(preds_)).sum().item()
            preds.extend(preds_.cpu().int().tolist())

    avg_loss = sum_loss / len(loader.dataset)
    accuracy = num_correct / len(loader.dataset)
    if return_preds:
        return avg_loss, accuracy, preds
    else:
        return avg_loss, accuracy


def get_model_and_optimizer(*, pretrained: bool = False) -> tuple[nn.Module, Optimizer]:
    """Crée un modèle et son optimiseur.

    Args:
        pretrained (bool, optionnel): Si True, charge les poids pré-entraînés sur ImageNet. Si False, initialise les
            poids aléatoirement. Par défaut, False. À noter que ce paramètre doit être spécifié par nom si on souhaite
            utiliser une valeur autre que la valeur par défaut.

    Returns:
        tuple[nn.Module, Optimizer]: le modèle et son optimiseur.
    """
    _seed_everything(42)
    torch.cuda.empty_cache()
    model = torchvision.models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)

    emb_size = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(emb_size, 4)
    torch.nn.init.kaiming_uniform_(model.classifier[-1].weight)
    model.classifier[-1].bias.data.fill_(0.0)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    return model, optimizer


def get_embeddings(model: nn.Module, loader: DataLoader) -> torch.Tensor:
    """Extrait les vecteurs latents (embeddings) du modèle pour toutes les images du dataloader.

    Args:
        model (nn.Module): le modèle à utiliser pour extraire les embeddings.
        loader (DataLoader): le dataloader contenant les images.

    Returns:
        torch.Tensor: un tenseur de forme (N, D) où N est le nombre d'images et D la dimension des embeddings.
    """
    emb = []

    def hook(module, input, output):
        emb.append(output.detach().cpu())

    handle = model.avgpool.register_forward_hook(hook)
    _ = validate(model, loader)
    handle.remove()
    return torch.cat(emb, dim=0).flatten(1)


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
