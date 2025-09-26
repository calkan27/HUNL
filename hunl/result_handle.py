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

