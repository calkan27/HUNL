"""
Thread-safe one-shot result container used by the asynchronous value server. A
ResultHandle exposes a set(value) method for producers and a blocking
result(as_numpy=True) for consumers, converting tensors to CPU/NumPy arrays when
requested.

Key class: ResultHandle. Methods: set (store a pair of tensors and release waiters),
result (block until available and return tensors or numpy arrays). Intended to be
created by a queue-owning worker and passed to requesters that need to await batched
model outputs.

Inputs: producer side provides a tuple of two tensors (player1, player2) already on a
desired device; consumer side sets as_numpy to control conversion. Outputs: either the
same tensors (no grad) or detached CPU NumPy arrays.

Invariants: single fulfillment per handle; waits until data is ready; preserves batch
dimension and order; no mutation of returned objects. Performance: minimal overhead
beyond an Event wait; conversion happens only when requested to keep hot paths on
device.
"""


import threading
import torch
class ResultHandle:
	def __init__(self):
		self._evt = threading.Event()
		self._out = None
	def set(self, value):
		self._out = value
		self._evt.set()
	def result(self, as_numpy: bool = True):
		self._evt.wait()
		v1, v2 = self._out
		if as_numpy:
			return v1.detach().cpu().numpy(), v2.detach().cpu().numpy()
		return v1, v2

