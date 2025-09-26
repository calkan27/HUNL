import random
import math
import torch
from torch import nn
from torch.optim import Adam

from cfr_solver import CFRSolver
from public_state import PublicState
from game_node import GameNode
from poker_utils import DECK


def train_flop_cfv(
	model,
	train_samples,
	val_samples,
	epochs: int = 200,
	batch_size: int = 1000,
	lr: float = 1e-3,
	lr_after: float = 1e-4,
	lr_drop_epoch: int = 150,
	weight_decay: float = 1e-6,
	device=None,
	seed=None,
	ckpt_dir=None,
	save_best: bool = True,
	target_provider=None,
	turn_model=None,
	turn_device=None,
	early_stop_patience: int = 30,
	min_delta: float = 0.0,
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

	if target_provider is None:
		if turn_model is not None:
			if turn_device is None:
				turn_device = device

			turn_model = turn_model.to(turn_device)
			turn_model.eval()

			def _tp(xb, y1b, y2b, tm):
				return default_turn_leaf_target_provider(
					xb.to(turn_device),
					y1b,
					y2b,
					tm,
				)

			target_provider = _tp

	optimizer = Adam(
		model.parameters(),
		lr=lr,
		weight_decay=float(weight_decay),
	)
	criterion = nn.SmoothL1Loss(reduction="mean")

	K = int(getattr(model, "num_clusters", 0))

	history = {
		"train_huber": [],
		"val_huber": [],
		"val_mae": [],
		"val_residual_max": [],
	}

	best_metric = None
	best_state = None
	pat = 0

	for e in range(int(epochs)):
		_maybe_step_lr(optimizer, e, lr_drop_epoch, lr_after)

		_train_one_epoch(
			model=model,
			train_samples=train_samples,
			batch_size=int(batch_size),
			device=device,
			K=K,
			criterion=criterion,
			optimizer=optimizer,
			target_provider=target_provider,
			turn_model=turn_model,
		)

		model.eval()

		with torch.no_grad():
			if target_provider is not None:
				def _prep_targets(samples):
					total_huber = 0.0
					total_mae = 0.0
					count = 0
					residual_max = 0.0

					n = len(samples)
					i = 0

					while i < n:
						j = min(i + int(batch_size), n)
						chunk = samples[i:j]
						i = j

						xb, y1b, y2b = _tensorize_batch(chunk, device)

						if turn_model is not None:
							t1, t2 = target_provider(xb, y1b, y2b, turn_model)
						else:
							t1, t2 = target_provider(xb, y1b, y2b, None)

						r1b, r2b = _ranges_from_inputs_inline(xb, K)
						p1, p2 = model(xb)
						f1, f2 = model.enforce_zero_sum(r1b, r2b, p1, p2)

						l1 = criterion(f1, t1)
						l2 = criterion(f2, t2)
						l = 0.5 * (l1 + l2)

						mae = 0.5 * (
							torch.mean(torch.abs(f1 - t1))
							+ torch.mean(torch.abs(f2 - t2))
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

				tr_huber, _, _ = _prep_targets(train_samples)

				if val_samples is not None:
					val_base = val_samples
				else:
					val_base = train_samples

				val_huber, val_mae, val_resmax = _prep_targets(val_base)
			else:
				tr_huber, _, _ = _epoch_eval(
					model=model,
					samples=train_samples,
					batch_size=int(batch_size),
					device=device,
					K=K,
					criterion=criterion,
					target_provider=None,
					turn_model=None,
				)

				if val_samples is not None:
					val_base = val_samples
				else:
					val_base = train_samples

				val_huber, val_mae, val_resmax = _epoch_eval(
					model=model,
					samples=val_base,
					batch_size=int(batch_size),
					device=device,
					K=K,
					criterion=criterion,
					target_provider=None,
					turn_model=None,
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
					_maybe_save_best(
						ckpt_dir=ckpt_dir,
						e=e,
						K=K,
						save_best=True,
						cur_metric=cur,
						best_metric=None,
						model=model,
						val_huber=val_huber,
						val_mae=val_mae,
					)

				pat = 0
			else:
				pat += 1

				if int(pat) >= int(early_stop_patience):
					break

		if ckpt_dir:
			_save_epoch_ckpt(
				ckpt_dir,
				e,
				K,
				model,
				val_huber,
				val_mae,
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

	return {
		"best_state": final_state,
		"history": history,
	}


def default_turn_leaf_target_provider(
	xb,
	y1b,
	y2b,
	turn_model,
):
	device_turn = next(turn_model.parameters()).device
	K = int(getattr(turn_model, "num_clusters", 0))
	B = 52

	n = xb.shape[0]
	t1_out = torch.zeros((n, K), dtype=torch.float32, device=xb.device)
	t2_out = torch.zeros((n, K), dtype=torch.float32, device=xb.device)

	def _decode_board(one_hot_vec):
		idx = (one_hot_vec > 0.5).nonzero(as_tuple=False).view(-1).tolist()
		out = []
		for j in idx:
			out.append(DECK[j])
		return out

	def _ranges_from_row(row):
		pn = float(row[0].item())
		board_vec = row[1 : 1 + B]
		r1 = row[1 + B : 1 + B + K].detach().cpu().tolist()
		r2 = row[1 + B + K : 1 + B + 2 * K].detach().cpu().tolist()
		return pn, _decode_board(board_vec.detach().cpu()), r1, r2

	total_initial = 400.0

	solver = CFRSolver(depth_limit=1, num_clusters=K)
	solver.models["turn"] = turn_model.to(device_turn).eval()
	solver.total_iterations = 1000

	i = 0
	while i < n:
		pn, board_cards, r1, r2 = _ranges_from_row(xb[i])

		scaled_pot = float(max(1e-6, min(1.0, pn)) * total_initial)

		ps = PublicState(
			initial_stacks=[200, 200],
			board_cards=list(board_cards),
			dealer=0,
		)
		ps.current_round = 1
		ps.current_bets = [0, 0]
		ps.pot_size = scaled_pot
		ps.last_raiser = None
		ps.stacks = [200, 200]
		ps.current_player = (ps.dealer + 1) % 2

		node = GameNode(ps)
		node.player_ranges[0] = {j: float(r1[j]) for j in range(K)}
		node.player_ranges[1] = {j: float(r2[j]) for j in range(K)}

		t1, t2 = solver.flop_label_targets_using_turn_net(node)

		j = 0
		while j < K:
			t1_out[i, j] = float(t1[j])
			t2_out[i, j] = float(t2[j])
			j += 1

		i += 1

	return t1_out.detach(), t2_out.detach()


def _prepare_targets(
	xb,
	y1b,
	y2b,
	target_provider,
	turn_model,
):
	if callable(target_provider):
		with torch.no_grad():
			return target_provider(xb, y1b, y2b, turn_model)
	return y1b, y2b


@torch.no_grad()
def _epoch_eval(
	model,
	samples,
	batch_size,
	device,
	K,
	criterion,
	target_provider,
	turn_model,
):
	model.eval()

	total_huber = 0.0
	total_mae = 0.0
	count = 0
	residual_max = 0.0

	n = len(samples)
	i = 0

	while i < n:
		j = min(i + int(batch_size), n)
		chunk = samples[i:j]
		i = j

		xb, y1b, y2b = _tensorize_batch(chunk, device)
		t1, t2 = _prepare_targets(xb, y1b, y2b, target_provider, turn_model)

		r1b, r2b = _ranges_from_inputs_inline(xb, K)
		p1, p2 = model(xb)
		f1, f2 = model.enforce_zero_sum(r1b, r2b, p1, p2)

		l1 = criterion(f1, t1)
		l2 = criterion(f2, t2)
		l = 0.5 * (l1 + l2)

		mae_1 = torch.mean(torch.abs(f1 - t1))
		mae_2 = torch.mean(torch.abs(f2 - t2))
		mae = 0.5 * (mae_1 + mae_2)

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


def _train_one_epoch(
	model,
	train_samples,
	batch_size,
	device,
	K,
	criterion,
	optimizer,
	target_provider,
	turn_model,
):
	model.train()

	n = len(train_samples)
	i = 0

	while i < n:
		j = min(i + int(batch_size), n)
		chunk = train_samples[i:j]
		i = j

		xb, y1b, y2b = _tensorize_batch(chunk, device)
		t1, t2 = _prepare_targets(xb, y1b, y2b, target_provider, turn_model)

		r1b, r2b = _ranges_from_inputs_inline(xb, K)
		p1, p2 = model(xb)
		f1, f2 = model.enforce_zero_sum(r1b, r2b, p1, p2)

		l1 = criterion(f1, t1)
		l2 = criterion(f2, t2)
		loss = 0.5 * (l1 + l2)

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()


def _maybe_step_lr(
	optimizer,
	epoch,
	lr_drop_epoch,
	lr_after,
):
	if epoch == int(lr_drop_epoch):
		new_lr = float(lr_after)
		for g in optimizer.param_groups:
			g["lr"] = new_lr


def _maybe_save_best(
	ckpt_dir,
	e,
	K,
	save_best,
	cur_metric,
	best_metric,
	model,
	val_huber,
	val_mae,
):
	best_state = None

	if save_best:
		if (best_metric is None) or (cur_metric < best_metric):
			best_metric = cur_metric
			best_state = {
				k: v.detach().cpu().clone()
				for k, v in model.state_dict().items()
			}

			if ckpt_dir:
				path = f"{ckpt_dir.rstrip('/')}/flop_cfv_best.pt"
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

	return best_metric, best_state


def _save_epoch_ckpt(
	ckpt_dir,
	e,
	K,
	model,
	val_huber,
	val_mae,
):
	if not ckpt_dir:
		return

	path_epoch = f"{ckpt_dir.rstrip('/')}/flop_cfv_epoch_{int(e)}.pt"
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


def _tensorize_batch(
	chunk,
	device,
):
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

	return xt, y1t, y2t


def _ranges_from_inputs_inline(
	x: torch.Tensor,
	K: int,
):
	sr1 = 1 + 52
	er1 = sr1 + int(K)
	sr2 = er1
	er2 = sr2 + int(K)
	return x[:, sr1:er1], x[:, sr2:er2]

