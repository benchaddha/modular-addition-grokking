import random
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import torch

from .config import Config
from .dataset import get_dataset
from .metrics import evaluate_accuracy
from .model import get_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_state_dict_from_checkpoint(checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def _train_for_grokking(
    cfg: Config,
    target_test_acc: float,
    max_epochs: int,
    eval_every: int,
) -> torch.nn.Module:
    _set_seed(cfg.train.seed)

    dataset = get_dataset(cfg)
    model = get_model(cfg, seed=cfg.train.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    train_tokens, train_labels = dataset.train_data()
    test_tokens, test_labels = dataset.test_data()
    train_size = train_tokens.shape[0]

    for epoch in range(max_epochs):
        model.train()
        indices = torch.randint(0, train_size, (cfg.train.batch_size,))
        batch_tokens = train_tokens[indices].to(model.cfg.device)
        batch_labels = train_labels[indices].to(model.cfg.device)

        logits = model(batch_tokens)[:, -1, :]
        loss = loss_fn(logits, batch_labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_eval = (epoch % eval_every == 0) or (epoch == max_epochs - 1)
        if not should_eval:
            continue

        test_acc = evaluate_accuracy(model, test_tokens, test_labels)
        print(
            f"[fourier-train] epoch={epoch} loss={loss.item():.4f} test_acc={test_acc:.4f}"
        )
        if test_acc >= target_test_acc:
            print(
                f"[fourier-train] reached test_acc>={target_test_acc:.2f} at epoch {epoch}"
            )
            break

    return model


def _embedding_power_spectrum(model: torch.nn.Module, p: int) -> np.ndarray:
    # Keep only number tokens 0..p-1, excluding "=" token.
    w_e_numbers = model.W_E.detach().float().cpu()[:p, :]
    fft_result = torch.fft.rfft(w_e_numbers, dim=0)
    power = (fft_result.abs() ** 2) / w_e_numbers.shape[0]
    # [freq, d_model] -> [d_model, freq] for heatmap (rows = dimensions)
    return power.T.numpy()


def _write_heatmap(
    spectrum: np.ndarray,
    p: int,
    output_path: Path,
) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=spectrum,
            x=np.arange(spectrum.shape[1]),
            y=np.arange(spectrum.shape[0]),
            colorscale="Viridis",
            colorbar=dict(title="Power"),
        )
    )
    fig.update_layout(
        title=f"The Harmonic Oscillator of AI: Embedding Spectrum (p={p})",
        xaxis_title="Frequency (k)",
        yaxis_title="Embedding Dimension",
        template="plotly_white",
        height=650,
        width=1050,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"[fourier] wrote heatmap to: {output_path}")


def visualize_fourier_spectrum(
    cfg: Config,
    checkpoint_path: Optional[Path] = None,
    output_path: Path = Path("results/figures/music_heatmap.html"),
    target_test_acc: float = 0.95,
    max_epochs: Optional[int] = None,
    eval_every: Optional[int] = None,
) -> Path:
    model = get_model(cfg, seed=cfg.train.seed)

    if checkpoint_path is not None:
        print(f"[fourier] loading checkpoint: {checkpoint_path}")
        model.load_state_dict(_load_state_dict_from_checkpoint(checkpoint_path))
    else:
        print("[fourier] no checkpoint provided; training fresh model for visualization")
        model = _train_for_grokking(
            cfg=cfg,
            target_test_acc=target_test_acc,
            max_epochs=max_epochs or cfg.train.epochs,
            eval_every=eval_every or cfg.train.eval_every,
        )

    spectrum = _embedding_power_spectrum(model=model, p=cfg.model.p)
    _write_heatmap(spectrum=spectrum, p=cfg.model.p, output_path=output_path)
    return output_path
