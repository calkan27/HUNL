"""
Turn-stage CFV trainer supporting both in-memory datasets and streaming datasets. It
provides a standard fit loop, streaming evaluation, and checkpointing, all on the
normalized fractions-of-pot target scale with per-sample zero-sum enforcement.

Key functions: train_turn_cfv and train_turn_cfv_streaming (fit loops), eval_stream and
_epoch_eval_list (evaluation for iterators and lists), batching helpers
(batcher_from_iter, _cfv_batcher), and slicing utilities (_slice_ranges). Metrics:
Huber, MAE, and maximum zero-sum residual; best weights are tracked and optionally
restored.

Inputs: model, sample lists or iterators yielding dicts {input_vector, target_v1,
target_v2}, epoch/batch budgets, learning-rate schedule, weight decay, device, seed, and
optional checkpoint directory. Outputs: dict with best_state and metric history;
optional files with epoch/best checkpoints carrying K and validation metrics.

Invariants: predictions are outer-adjusted before loss; inputs are sliced consistently;
device transfers are explicit and minimal; history records are numeric and
JSON-serializable. Performance: streaming mode reduces memory pressure and pairs well
with on-the-fly generation.
"""


from hunl.constants import EPS_ZS
import random
import torch
from torch import nn
from torch.optim import Adam


def batcher_from_iter(samples_iter, bs, device):
	buf_x, buf_y1, buf_y2 = [], [], []

	for rec in samples_iter:
		buf_x.append(rec["input_vector"])
		buf_y1.append(rec["target_v1"])
		buf_y2.append(rec["target_v2"])

		if len(buf_x) >= bs:
			xt = torch.tensor(
			 buf_x,
			 dtype=torch.float32,
			 device=device,
			)
			y1t = torch.tensor(
			 buf_y1,
			 dtype=torch.float32,
			 device=device,
			)
			y2t = torch.tensor(
			 buf_y2,
			 dtype=torch.float32,
			 device=device,
			)
			yield xt, y1t, y2t
			buf_x, buf_y1, buf_y2 = [], [], []

	if buf_x:
		xt = torch.tensor(buf_x, dtype=torch.float32, device=device)
		y1t = torch.tensor(buf_y1, dtype=torch.float32, device=device)
		y2t = torch.tensor(buf_y2, dtype=torch.float32, device=device)
		yield xt, y1t, y2t


def _slice_ranges(xb, K: int):
	sr1 = 1 + 52
	er1 = sr1 + int(K)
	sr2 = er1
	er2 = sr2 + int(K)
	return xb[:, sr1:er1], xb[:, sr2:er2]


def eval_stream(model, samples_iter, batch_size, device, K, criterion):
	model.eval()

	total_huber = 0.0
	total_mae = 0.0
	count = 0
	residual_max = 0.0

	with torch.no_grad():
		for xb, y1b, y2b in batcher_from_iter(
		 samples_iter,
		 int(batch_size),
		 device,
		):
			r1b, r2b = _slice_ranges(xb, K)

			p1, p2 = model(xb)
			f1, f2 = model.enforce_zero_sum(r1b, r2b, p1, p2)

			l1 = criterion(f1, y1b)
			l2 = criterion(f2, y2b)
			l = 0.5 * (l1 + l2)

			mae = 0.5 * (
			 torch.mean(torch.abs(f1 - y1b))
			 + torch.mean(torch.abs(f2 - y2b))
			)

			s1 = torch.sum(r1b * f1, dim=1)
			s2 = torch.sum(r2b * f2, dim=1)
			res = torch.abs(s1 + s2)

			bs = xb.shape[0]
			total_huber += float(l.item()) * bs
			total_mae += float(mae.item()) * bs
			count += bs

			if res.numel() > 0:
				mx = float(torch.max(res).item())
			else:
				mx = 0.0

			if mx > residual_max:
				residual_max = mx

	den = max(1, count)
	return total_huber / den, total_mae / den, residual_max


def _epoch_eval_list(model, samples, batch_size, device, K, criterion):
	model.eval()

	total_huber = 0.0
	total_mae = 0.0
	count = 0
	residual_max = 0.0

	with torch.no_grad():
		for xb, y1b, y2b in _cfv_batcher(
		 samples,
		 int(batch_size),
		 shuffle=False,
		 device=device,
		):
			r1b, r2b = _slice_ranges(xb, K)

			p1, p2 = model(xb)
			f1, f2 = model.enforce_zero_sum(r1b, r2b, p1, p2)

			l1 = criterion(f1, y1b)
			l2 = criterion(f2, y2b)
			l = 0.5 * (l1 + l2)

			mae = 0.5 * (
			 torch.mean(torch.abs(f1 - y1b))
			 + torch.mean(torch.abs(f2 - y2b))
			)

			s1 = torch.sum(r1b * f1, dim=1)
			s2 = torch.sum(r2b * f2, dim=1)
			res = torch.abs(s1 + s2)

			bs = xb.shape[0]
			total_huber += float(l.item()) * bs
			total_mae += float(mae.item()) * bs
			count += bs

			if res.numel() > 0:
				mx = float(torch.max(res).item())
			else:
				mx = 0.0

			if mx > residual_max:
				residual_max = mx

	den = max(1, count)
	return total_huber / den, total_mae / den, residual_max


def train_turn_cfv(
 model,
 train_samples,
 val_samples,
 epochs: int = 200,
 batch_size: int = 1000,
 lr: float = 1e-3,
 lr_after: float = 1e-4,
 lr_drop_epoch: int = 150,
 weight_decay: float = EPS_ZS,
 device=None,
 seed=None,
 ckpt_dir=None,
 save_best: bool = True,
):
	if seed is not None:
		rseed = int(seed)
		random.seed(rseed)
		torch.manual_seed(rseed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(rseed)

	if device is None:
		if torch.cuda.is_available():
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")

	model = model.to(device)
	optimizer = Adam(
	 model.parameters(),
	 lr=lr,
	 weight_decay=float(weight_decay),
	)
	criterion = nn.SmoothL1Loss(reduction="mean")

	K = int(getattr(model, "num_clusters", 0))

	best_metric = None
	best_state = None

	history = {
	 "train_huber": [],
	 "val_huber": [],
	 "val_mae": [],
	 "val_residual_max": [],
	}

	for e in range(int(epochs)):
		if e == int(lr_drop_epoch):
			for g in optimizer.param_groups:
				g["lr"] = float(lr_after)

		model.train()

		for xb, y1b, y2b in _cfv_batcher(
		 train_samples,
		 int(batch_size),
		 shuffle=True,
		 device=device,
		):
			r1b, r2b = _slice_ranges(xb, K)

			p1, p2 = model(xb)
			f1, f2 = model.enforce_zero_sum(r1b, r2b, p1, p2)

			l1 = criterion(f1, y1b)
			l2 = criterion(f2, y2b)
			loss = 0.5 * (l1 + l2)

			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

		tr_huber, _, _ = _epoch_eval_list(
		 model,
		 train_samples,
		 batch_size,
		 device,
		 K,
		 criterion,
		)

		if val_samples is not None:
			val_set = val_samples
		else:
			val_set = train_samples

		val_huber, val_mae, val_resmax = _epoch_eval_list(
		 model,
		 val_set,
		 batch_size,
		 device,
		 K,
		 criterion,
		)

		history["train_huber"].append(float(tr_huber))
		history["val_huber"].append(float(val_huber))
		history["val_mae"].append(float(val_mae))
		history["val_residual_max"].append(float(val_resmax))

		cur = float(val_huber)

		if save_best:
			if (best_metric is None) or (cur < best_metric):
				best_metric = cur
				best_state = {
				 k: v.detach().cpu().clone()
				 for k, v in model.state_dict().items()
				}

				if ckpt_dir:
					path = f"{ckpt_dir.rstrip('/')}/turn_cfv_best.pt"
					torch.save(
					 {
					  "epoch": int(e),
					  "val_huber": float(val_huber),
					  "val_mae": float(val_mae),
					  "state_dict": best_state,
					  "num_clusters": int(K),
					 },
					 path,
					)

		if ckpt_dir:
			path_epoch = (
			 f"{ckpt_dir.rstrip('/')}/turn_cfv_epoch_{int(e)}.pt"
			)
			torch.save(
			 {
			  "epoch": int(e),
			  "val_huber": float(val_huber),
			  "val_mae": float(val_mae),
			  "state_dict": {
			   k: v.detach().cpu()
			   for k, v in model.state_dict().items()
			  },
			  "num_clusters": int(K),
			 },
			 path_epoch,
			)

	if save_best:
		if best_state is not None:
			model.load_state_dict(best_state)

	if best_state is not None:
		final_state = best_state
	else:
		final_state = {
		 k: v.detach().cpu().clone()
		 for k, v in model.state_dict().items()
		}

	return {"best_state": final_state, "history": history}


def train_turn_cfv_streaming(
 model,
 train_iter,
 val_iter=None,
 epochs: int = 200,
 batch_size: int = 1000,
 lr: float = 1e-3,
 lr_after: float = 1e-4,
 lr_drop_epoch: int = 150,
 weight_decay: float = EPS_ZS,
 device=None,
 seed=None,
 ckpt_dir=None,
 save_best: bool = True,
):
	if seed is not None:
		rseed = int(seed)
		random.seed(rseed)
		torch.manual_seed(rseed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(rseed)

	if device is None:
		if torch.cuda.is_available():
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")

	model = model.to(device)
	optimizer = Adam(
	 model.parameters(),
	 lr=lr,
	 weight_decay=float(weight_decay),
	)
	criterion = nn.SmoothL1Loss(reduction="mean")

	K = int(getattr(model, "num_clusters", 0))

	best_metric = None
	best_state = None

	history = {"train_huber": [], "val_huber": [], "val_mae": []}

	for e in range(int(epochs)):
		if e == int(lr_drop_epoch):
			for g in optimizer.param_groups:
				g["lr"] = float(lr_after)

		model.train()

		for xb, y1b, y2b in batcher_from_iter(
		 train_iter(),
		 int(batch_size),
		 device,
		):
			r1b, r2b = _slice_ranges(xb, K)

			p1, p2 = model(xb)
			f1, f2 = model.enforce_zero_sum(r1b, r2b, p1, p2)

			l1 = criterion(f1, y1b)
			l2 = criterion(f2, y2b)
			loss = 0.5 * (l1 + l2)

			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

		tr_huber, _tr_mae, _tr_res = eval_stream(
		 model,
		 train_iter(),
		 batch_size,
		 device,
		 K,
		 criterion,
		)

		if val_iter is not None:
			val_huber, val_mae, _val_res = eval_stream(
			 model,
			 val_iter(),
			 batch_size,
			 device,
			 K,
			 criterion,
			)
		else:
			val_huber = tr_huber
			val_mae = 0.0

		history["train_huber"].append(float(tr_huber))
		history["val_huber"].append(float(val_huber))
		history["val_mae"].append(float(val_mae))

		cur = float(val_huber)

		if save_best:
			if (best_metric is None) or (cur < best_metric):
				best_metric = cur
				best_state = {
				 k: v.detach().cpu().clone()
				 for k, v in model.state_dict().items()
				}

				if ckpt_dir:
					path = f"{ckpt_dir.rstrip('/')}/turn_cfv_best.pt"
					torch.save(
					 {
					  "epoch": int(e),
					  "val_huber": float(val_huber),
					  "val_mae": float(val_mae),
					  "state_dict": best_state,
					  "num_clusters": int(K),
					 },
					 path,
					)

		if ckpt_dir:
			path_epoch = (
			 f"{ckpt_dir.rstrip('/')}/turn_cfv_epoch_{int(e)}.pt"
			)
			torch.save(
			 {
			  "epoch": int(e),
			  "val_huber": float(val_huber),
			  "val_mae": float(val_mae),
			  "state_dict": {
			   k: v.detach().cpu()
			   for k, v in model.state_dict().items()
			  },
			  "num_clusters": int(K),
			 },
			 path_epoch,
			)

	if save_best:
		if best_state is not None:
			model.load_state_dict(best_state)

	if best_state is not None:
		final_state = best_state
	else:
		final_state = {
		 k: v.detach().cpu().clone()
		 for k, v in model.state_dict().items()
		}

	return {"best_state": final_state, "history": history}


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

		x, y1, y2 = [], [], []

		for s in chunk:
			x.append(s["input_vector"])
			y1.append(s["target_v1"])
			y2.append(s["target_v2"])

		xt = torch.tensor(x, dtype=torch.float32, device=device)
		y1t = torch.tensor(y1, dtype=torch.float32, device=device)
		y2t = torch.tensor(y2, dtype=torch.float32, device=device)

		yield xt, y1t, y2t

