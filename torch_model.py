from __future__ import annotations

import copy
import random

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TabularMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=(64, 32), dropout=0.2):
        super().__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(current_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_size = hidden_size
        layers.append(nn.Linear(current_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features).squeeze(1)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """A reproducible sklearn-compatible PyTorch classifier for tabular data."""

    def __init__(
        self,
        hidden_sizes=(64, 32),
        dropout=0.2,
        learning_rate=0.001,
        weight_decay=0.0001,
        epochs=40,
        batch_size=512,
        validation_size=0.15,
        patience=6,
        random_state=42,
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.patience = patience
        self.random_state = random_state

    def _set_seed(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def fit(self, features, target):
        self._set_seed()
        values = np.asarray(features, dtype=np.float32)
        labels = np.asarray(target, dtype=np.float32)
        train_x, validation_x, train_y, validation_y = train_test_split(
            values,
            labels,
            test_size=self.validation_size,
            random_state=self.random_state,
            stratify=labels,
        )
        self.classes_ = np.array([0, 1])
        self.model_ = TabularMLP(values.shape[1], self.hidden_sizes, self.dropout)
        positive_weight = (train_y == 0).sum() / max((train_y == 1).sum(), 1)
        loss_function = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([positive_weight], dtype=torch.float32)
        )
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        validation_features = torch.from_numpy(validation_x)
        validation_target = torch.from_numpy(validation_y)
        best_state = copy.deepcopy(self.model_.state_dict())
        best_loss = float("inf")
        stale_epochs = 0
        self.history_ = {"train_loss": [], "validation_loss": []}
        for _ in range(self.epochs):
            self.model_.train()
            batch_losses = []
            for batch_features, batch_target in train_loader:
                optimizer.zero_grad()
                loss = loss_function(self.model_(batch_features), batch_target)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            self.model_.eval()
            with torch.no_grad():
                validation_loss = loss_function(
                    self.model_(validation_features), validation_target
                ).item()
            self.history_["train_loss"].append(float(np.mean(batch_losses)))
            self.history_["validation_loss"].append(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, features):
        self.model_.eval()
        values = torch.from_numpy(np.asarray(features, dtype=np.float32))
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model_(values)).numpy()
        return np.column_stack([1 - probabilities, probabilities])

    def predict(self, features):
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)
