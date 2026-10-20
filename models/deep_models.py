"""Deep learning models based on Wav2Vec2 for Voice Biomarker Platform v2.

Provides ``Wav2Vec2Classifier`` -- a fine-tunable speech classifier built on
HuggingFace Transformers.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import Wav2Vec2Model

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class Wav2Vec2Classifier(nn.Module):
    """Wav2Vec2-based audio classifier.

    Architecture::

        Wav2Vec2 base -> mean pooling -> dropout -> linear -> ReLU -> linear

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 2).
    model_name : str
        HuggingFace model identifier (default ``"facebook/wav2vec2-base"``).
    freeze_base : bool
        If *True* the Wav2Vec2 backbone weights are frozen during training.
    dropout : float
        Dropout probability before the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "facebook/wav2vec2-base",
        freeze_base: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The 'transformers' package is required for Wav2Vec2Classifier. "
                "Install it with: pip install transformers"
            )

        self._num_classes = num_classes
        self._model_name = model_name
        self._freeze_base = freeze_base

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size = self.wav2vec2.config.hidden_size

        if freeze_base:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(batch, sequence_length)``
            Raw waveform or pre-processed audio tensor.

        Returns
        -------
        Tensor of shape ``(batch, num_classes)``
            Raw logits.
        """
        outputs = self.wav2vec2(x)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        logits = self.classifier_head(pooled)
        return logits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 1e-4,
        patience: int = 10,
        device: Optional[str] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping and LR scheduling.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        epochs : int
        lr : float
        patience : int
            Epochs without improvement before stopping early.
        device : str, optional
            ``"cuda"`` / ``"cpu"``.  Auto-detected when *None*.
        class_weights : Tensor, optional
            Weight tensor for cross-entropy loss (useful for imbalanced data).

        Returns
        -------
        dict
            History with keys ``"train_loss"``, ``"val_loss"``,
            ``"train_acc"``, ``"val_acc"``.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        criterion = (
            nn.CrossEntropyLoss(weight=class_weights.to(device))
            if class_weights is not None
            else nn.CrossEntropyLoss()
        )

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience // 2,
        )

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, Any]] = None
        no_improve = 0

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            # ---- Train ----
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = self.forward(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total += batch_x.size(0)

            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)

            # ---- Validate ----
            val_loss, val_acc = self._evaluate_loop(val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            scheduler.step(val_loss)

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch, epochs, train_loss, train_acc, val_loss, val_acc,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        return history

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class on a single audio waveform or batch.

        Parameters
        ----------
        audio : ndarray or Tensor
            1-D waveform for single sample, or 2-D ``(batch, length)``.

        Returns
        -------
        ndarray of int
            Predicted class labels.
        """
        device = next(self.parameters()).device
        self.eval()

        if isinstance(audio, np.ndarray):
            tensor = torch.tensor(audio, dtype=torch.float32)
        else:
            tensor = audio.float()

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device)
        logits = self.forward(tensor)
        return logits.argmax(dim=1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Return softmax class probabilities."""
        device = next(self.parameters()).device
        self.eval()

        if isinstance(audio, np.ndarray):
            tensor = torch.tensor(audio, dtype=torch.float32)
        else:
            tensor = audio.float()

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device)
        logits = self.forward(tensor)
        return torch.softmax(logits, dim=1).cpu().numpy()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_loader: DataLoader,
        device: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate on a test DataLoader.

        Returns
        -------
        dict
            Keys: ``"loss"``, ``"accuracy"``, ``"f1_macro"``.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        criterion = nn.CrossEntropyLoss()
        loss, acc, all_preds, all_labels = self._evaluate_loop_full(
            test_loader, criterion, device,
        )

        from sklearn.metrics import f1_score

        f1 = float(f1_score(all_labels, all_preds, average="macro"))
        return {"loss": loss, "accuracy": acc, "f1_macro": f1}

    # ------------------------------------------------------------------
    # Fine-tuning support
    # ------------------------------------------------------------------

    def unfreeze_base(self, num_layers: Optional[int] = None) -> None:
        """Unfreeze Wav2Vec2 base layers for fine-tuning.

        Parameters
        ----------
        num_layers : int, optional
            If provided, unfreeze only the last *num_layers* encoder layers.
            If *None*, unfreeze the entire base.
        """
        if num_layers is not None:
            encoder_layers = self.wav2vec2.encoder.layer
            total = len(encoder_layers)
            start = max(0, total - num_layers)
            for i, layer in enumerate(encoder_layers):
                requires_grad = i >= start
                for param in layer.parameters():
                    param.requires_grad = requires_grad
            logger.info("Unfroze last %d of %d encoder layers", num_layers, total)
        else:
            for param in self.wav2vec2.parameters():
                param.requires_grad = True
            logger.info("Unfroze entire Wav2Vec2 base")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save model state dict and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": {
                    "num_classes": self._num_classes,
                    "model_name": self._model_name,
                    "freeze_base": self._freeze_base,
                },
            },
            path,
        )
        logger.info("Model saved to %s", path)

    @classmethod
    def load(
        cls, path: Union[str, Path], device: Optional[str] = None,
    ) -> Wav2Vec2Classifier:
        """Load model from checkpoint."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = cls(
            num_classes=config["num_classes"],
            model_name=config["model_name"],
            freeze_base=config["freeze_base"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_loop(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        device: str,
    ) -> Tuple[float, float]:
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = self.forward(batch_x)
                loss = criterion(logits, batch_y)
                running_loss += loss.item() * batch_x.size(0)
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total += batch_x.size(0)
        return running_loss / max(total, 1), correct / max(total, 1)

    def _evaluate_loop_full(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        device: str,
    ) -> Tuple[float, float, List[int], List[int]]:
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds: List[int] = []
        all_labels: List[int] = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = self.forward(batch_x)
                loss = criterion(logits, batch_y)
                running_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch_y.cpu().tolist())
        return (
            running_loss / max(total, 1),
            correct / max(total, 1),
            all_preds,
            all_labels,
        )
