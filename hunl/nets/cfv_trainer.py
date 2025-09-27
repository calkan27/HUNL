from hunl.constants import EPS_ZS
import math
import random
import sys
import torch
from torch import nn
from torch.optim import Adam

try:
	from utils import set_global_seed
except Exception:
	def set_global_seed(seed: int) -> None:
		random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)


def _get_eval_loss_hook(default_fn):
	if "cfv_trainer" in sys.modules:
		mod = sys.modules["cfv_trainer"]
		fn = getattr(mod, "_eval_loss_cfv", None)
		if callable(fn):
			return fn
	return default_fn


def train_cfv_network(
 model,
 train_samples,
 val_samples,
 epochs=350,
 batch_size=1000,
 lr=1e-3,
 lr_drop_epoch=200,
 lr_after=1e-4,
 weight_decay=EPS_ZS,
 device=None,
 seed=None,
 early_stop_patience=30,
 min_delta=0.0,
):
	if seed is not None:
		set_global_seed(int(seed))
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = model.to(device)
	optimizer = Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay))
	criterion = nn.SmoothL1Loss(reduction="mean")
	K = int(getattr(model, "num_clusters", 0))

	best_state = None
	best_val = math.inf
	history = {
	 "train_huber": [],
	 "train_mae": [],
	 "train_residual_max": [],
	 "val_huber": [],
	 "val_mae": [],
	 "val_residual_max": [],
	}
	pat = 0
	eval_loss_fn = _get_eval_loss_hook(_eval_loss_cfv)

	for e in range(int(epochs)):
		if e == int(lr_drop_epoch):
			for g in optimizer.param_groups:
				g["lr"] = float(lr_after)

		model.train()
		for xb, y1b, y2b in _cfv_batcher(train_samples, int(batch_size), shuffle=True, device=device):
			r1b, r2b = _ranges_from_inputs(xb, K)
			p1, p2 = model(xb)
			f1, f2 = model.enforce_zero_sum(r1b, r2b, p1, p2)
			l1 = criterion(f1, y1b)
			l2 = criterion(f2, y2b)
			loss = 0.5 * (l1 + l2)
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

		tr_huber, tr_mae, tr_resmax = eval_loss_fn(
		 model=model,
		 train_samples=train_samples,
		 val_samples=val_samples,
		 split="train",
		 batch_size=int(batch_size),
		 device=device,
		 criterion=criterion,
		 K=K,
		)
		val_huber, val_mae, val_resmax = eval_loss_fn(
		 model=model,
		 train_samples=train_samples,
		 val_samples=val_samples,
		 split="val",
		 batch_size=int(batch_size),
		 device=device,
		 criterion=criterion,
		 K=K,
		)

		history["train_huber"].append(float(tr_huber))
		history["train_mae"].append(float(tr_mae))
		history["train_residual_max"].append(float(tr_resmax))
		history["val_huber"].append(float(val_huber))
		history["val_mae"].append(float(val_mae))
		history["val_residual_max"].append(float(val_resmax))

		if float(val_huber) + float(min_delta) < float(best_val):
			best_val = float(val_huber)
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
			pat = 0
		else:
			pat += 1
			if pat >= int(early_stop_patience):
				break

	return {"best_state": best_state, "best_val_loss": float(best_val), "history": history}


def _cfv_batcher(samples, batch_size, shuffle, device):
	n = len(samples)
	idx = list(range(n))
	if shuffle:
		random.shuffle(idx)

	i = 0
	while i < n:
		j = min(i + batch_size, n)
		chunk = [samples[k] for k in idx[i:j]]
		i = j

		x = []
		y1 = []
		y2 = []
		for s in chunk:
			x.append(s["input_vector"])
			y1.append(s["target_v1"])
			y2.append(s["target_v2"])

		xt = torch.tensor(x, dtype=torch.float32, device=device)
		y1t = torch.tensor(y1, dtype=torch.float32, device=device)
		y2t = torch.tensor(y2, dtype=torch.float32, device=device)
		yield xt, y1t, y2t


def _ranges_from_inputs(x: torch.Tensor, K: int):
	sr1 = 1 + 52
	er1 = sr1 + K
	sr2 = er1
	er2 = sr2 + K
	return x[:, sr1:er1], x[:, sr2:er2]


def _eval_loss_cfv(model, train_samples, val_samples, split, batch_size, device, criterion, K: int):
	model.eval()
	total_huber = 0.0
	total_mae = 0.0
	count = 0
	residual_max = 0.0

	with torch.no_grad():
		source = val_samples if split == "val" else train_samples
		for xb, y1b, y2b in _cfv_batcher(source, batch_size, shuffle=False, device=device):
			r1b, r2b = _ranges_from_inputs(xb, K)
			p1, p2 = model(xb)
			f1, f2 = model.enforce_zero_sum(r1b, r2b, p1, p2)
			l1 = criterion(f1, y1b)
			l2 = criterion(f2, y2b)
			l = 0.5 * (l1 + l2)
			mae = 0.5 * (torch.mean(torch.abs(f1 - y1b)) + torch.mean(torch.abs(f2 - y2b)))
			s1 = torch.sum(r1b * f1, dim=1)
			s2 = torch.sum(r2b * f2, dim=1)
			res = torch.abs(s1 + s2)

			bs = xb.shape[0]
			total_huber += float(l.item()) * bs
			total_mae += float(mae.item()) * bs
			count += bs

			mx = float(torch.max(res).item()) if res.numel() > 0 else 0.0
			if mx > residual_max:
				residual_max = mx

	den = max(1, count)
	return total_huber / den, total_mae / den, residual_max

