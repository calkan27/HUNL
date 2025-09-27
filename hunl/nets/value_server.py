import threading
import queue
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
from hunl.utils.result_handle import ResultHandle


class ValueServer:
	def __init__(
		self,
		models: Dict[str, torch.nn.Module],
		device: Optional[torch.device] = None,
		max_batch_size: int = 1024,
		max_wait_ms: int = 2,
	):
		self.models: Dict[str, torch.nn.Module] = {
			str(k): v for k, v in dict(models).items()
		}

		if device is not None:
			self.device = device
		else:
			if torch.cuda.is_available():
				self.device = torch.device("cuda")
			else:
				self.device = torch.device("cpu")

		for k in list(self.models.keys()):
			self.models[k] = self.models[k].to(self.device)
			self.models[k].eval()

		self.max_batch: int = int(max_batch_size)
		self.max_wait_ms: int = int(max_wait_ms)

		self._q: "queue.Queue[Tuple[str, torch.Tensor, bool, Optional[float], ResultHandle]]" = queue.Queue()
		self._stop = threading.Event()
		self._thr: Optional[threading.Thread] = None

		self._counters: Dict[str, int] = {
			"preflop": 0,
			"flop": 0,
			"turn": 0,
			"river": 0,
		}

		self._resid_stats: Dict[str, Dict[str, float]] = {
			"overall": {"max": 0.0, "sum": 0.0, "count": 0.0},
			"preflop": {"max": 0.0, "sum": 0.0, "count": 0.0},
			"flop": {"max": 0.0, "sum": 0.0, "count": 0.0},
			"turn": {"max": 0.0, "sum": 0.0, "count": 0.0},
			"river": {"max": 0.0, "sum": 0.0, "count": 0.0},
		}

		self.total_initial_default: float = 400.0



	@staticmethod
	def _queue_has_items(q: "queue.Queue") -> bool:
		if hasattr(q, "qsize"):
			return q.qsize() > 0
		else:
			return False

	@classmethod
	def _queue_get_nowait(cls, q: "queue.Queue"):
		if cls._queue_has_items(q):
			return True, q.get_nowait()
		else:
			return False, None

	@staticmethod
	def _concat_on_device(
		tensors: List[torch.Tensor],
		device: torch.device,
	) -> torch.Tensor:
		if len(tensors) > 0:
			return torch.cat([t.to(device) for t in tensors], dim=0)
		else:
			return torch.zeros((0, 1), device=device)

	@staticmethod
	def _range_slices_from_batch(
		batch: torch.Tensor,
		model: torch.nn.Module,
	) -> Tuple[slice, slice]:
		K = int(getattr(model, "num_clusters", 0))
		insz = int(getattr(model, "input_size", int(batch.shape[1])))
		board_dim = max(0, insz - (1 + 2 * K))

		start_r1 = 1 + board_dim
		end_r1 = start_r1 + K
		start_r2 = end_r1
		end_r2 = start_r2 + K

		return slice(start_r1, end_r1), slice(start_r2, end_r2)

	@staticmethod
	def _split_offsets(
		idx_list: List[int],
		xs: List[torch.Tensor],
	) -> List[int]:
		offsets: List[int] = []
		acc = 0

		for i in idx_list:
			offsets.append(acc)
			acc += int(xs[i].shape[0])

		return offsets

	@staticmethod
	def _maybe_scale_outputs(
		out_pair: Tuple[torch.Tensor, torch.Tensor],
		x: torch.Tensor,
		scale_flag: bool,
		total_initial: Optional[float],
		total_initial_default: float,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		out1, out2 = out_pair

		if scale_flag:
			vec = x.detach().cpu().view(x.shape[0], -1)

			if vec.shape[1] > 0:
				pot_norm = float(vec[0, 0].item())
			else:
				pot_norm = 0.0

			if total_initial is not None:
				ti = float(total_initial)

				if ti <= 0.0:
					ti = float(total_initial_default)
				else:
					ti = ti
			else:
				ti = float(total_initial_default)

			scale = pot_norm * ti
			return out1 * scale, out2 * scale
		else:
			return out1, out2

	@staticmethod
	def _update_residual_stats(
		res: torch.Tensor,
		stats: Dict[str, Dict[str, float]],
		stage: str,
	) -> None:
		mx = float(torch.max(res).item()) if int(res.numel()) > 0 else 0.0
		sm = float(torch.sum(res).item()) if int(res.numel()) > 0 else 0.0
		ct = float(res.numel())

		stats_stage = stats.setdefault(
			str(stage),
			{"max": 0.0, "sum": 0.0, "count": 0.0},
		)
		stats_over = stats.setdefault(
			"overall",
			{"max": 0.0, "sum": 0.0, "count": 0.0},
		)

		if mx > stats_stage["max"]:
			stats_stage["max"] = mx
		else:
			pass

		stats_stage["sum"] += sm
		stats_stage["count"] += ct

		if mx > stats_over["max"]:
			stats_over["max"] = mx
		else:
			pass

		stats_over["sum"] += sm
		stats_over["count"] += ct



	def start(self) -> None:
		if self._thr is not None:
			if self._thr.is_alive():
				return
			else:
				pass
		else:
			pass

		self._stop.clear()

		self._thr = threading.Thread(
			target=self._run,
			name="ValueServerWorker",
			daemon=True,
		)
		self._thr.start()

	def stop(self, join: bool = True) -> None:
		self._stop.set()

		if join:
			if self._thr is not None:
				self._thr.join(timeout=1.0)
			else:
				pass
		else:
			pass

	def query(
		self,
		stage: str,
		inputs: torch.Tensor,
		scale_to_pot: bool = False,
		as_numpy: bool = True,
		total_initial: Optional[float] = None,
	) -> Tuple[Any, Any]:
		if int(inputs.dim()) == 1:
			inputs = inputs.unsqueeze(0)
		else:
			pass

		self.start()

		h = ResultHandle()

		self._q.put(
			(
				str(stage),
				inputs.detach().clone(),
				bool(scale_to_pot),
				total_initial,
				h,
			)
		)

		return h.result(as_numpy=as_numpy)

	def query_many(
		self,
		stage: str,
		batch_inputs: torch.Tensor,
		scale_to_pot: bool = False,
		as_numpy: bool = True,
		total_initial: Optional[float] = None,
	) -> Tuple[Any, Any]:
		return self.query(
			stage,
			batch_inputs,
			scale_to_pot=scale_to_pot,
			as_numpy=as_numpy,
			total_initial=total_initial,
		)

	def get_counters(self) -> Dict[str, int]:
		out: Dict[str, int] = {}

		for k, v in self._counters.items():
			out[str(k)] = int(v)

		return out

	# ---------------------------
	# Worker internals
	# ---------------------------

	def _pull_one_request(self):
		ok, item = self._queue_get_nowait(self._q)

		if ok:
			return True, item
		else:
			time.sleep(0.001)
			ok2, item2 = self._queue_get_nowait(self._q)

			if ok2:
				return True, item2
			else:
				return False, None

	def _run(self) -> None:
		while not self._stop.is_set():
			ok, item = self._pull_one_request()

			if ok:
				stage0, x0, scale0, t0, h0 = item
			else:
				continue

			stages: List[str] = [stage0]
			xs: List[torch.Tensor] = [x0]
			scale_flags: List[bool] = [scale0]
			totals: List[Optional[float]] = [t0]
			handles: List[ResultHandle] = [h0]

			deadline = time.time() + (self.max_wait_ms / 1000.0)

			while len(xs) < int(self.max_batch):
				if time.time() < deadline:
					ok2, item2 = self._queue_get_nowait(self._q)

					if ok2:
						stg, xt, sc, tt, hh = item2
						stages.append(stg)
						xs.append(xt)
						scale_flags.append(sc)
						totals.append(tt)
						handles.append(hh)
					else:
						break
				else:
					break

			self._process_batch(
				stages,
				xs,
				scale_flags,
				totals,
				handles,
			)

	def _process_batch(
		self,
		stages: List[str],
		xs: List[torch.Tensor],
		scale_flags: List[bool],
		totals: List[Optional[float]],
		handles: List[ResultHandle],
	) -> None:
		idx_by_stage: Dict[str, List[int]] = {}

		i = 0
		while i < len(stages):
			st = stages[i]

			if st in idx_by_stage:
				idx_by_stage[st].append(i)
			else:
				idx_by_stage[st] = [i]

			i += 1

		for st, idxs in idx_by_stage.items():
			model = self.models.get(str(st))

			if (model is None) or (len(idxs) == 0):
				for i2 in idxs:
					handles[i2].set(
						(torch.zeros(1, 0), torch.zeros(1, 0))
					)
				continue
			else:
				with torch.no_grad():
					batch = self._concat_on_device(
						[xs[i2] for i2 in idxs],
						self.device,
					)

					r1_sl, r2_sl = self._range_slices_from_batch(
						batch,
						model,
					)

					r1 = batch[:, r1_sl]
					r2 = batch[:, r2_sl]

					p1, p2 = model(batch)
					f1, f2 = model.enforce_zero_sum(r1, r2, p1, p2)

					self._counters[str(st)] = int(
						self._counters.get(str(st), 0)
						+ int(batch.shape[0])
					)

					s1 = torch.sum(r1 * f1, dim=1)
					s2 = torch.sum(r2 * f2, dim=1)
					res = torch.abs(s1 + s2).detach().cpu()

					self._update_residual_stats(
						res,
						self._resid_stats,
						str(st),
					)

					offsets = self._split_offsets(idxs, xs)

					j = 0
					while j < len(idxs):
						i2 = idxs[j]
						start = int(offsets[j])
						count = int(xs[i2].shape[0])

						out1 = f1[start : start + count, :].clone()
						out2 = f2[start : start + count, :].clone()

						out1, out2 = self._maybe_scale_outputs(
							(out1, out2),
							xs[i2],
							bool(scale_flags[i2]),
							totals[i2],
							float(self.total_initial_default),
						)

						handles[i2].set((out1, out2))
						j += 1

