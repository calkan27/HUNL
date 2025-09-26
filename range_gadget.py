class RangeGadget:
	def __init__(self):
		self.upper = {}
	def begin(self, initial_upper=None):
		if isinstance(initial_upper, dict):
			for k, v in initial_upper.items():
				self.upper[int(k)] = float(v)
		return dict(self.upper)
	def update(self, observed_upper):
		if not isinstance(observed_upper, dict):
			return dict(self.upper)
		for k, v in observed_upper.items():
			kk = int(k)
			fv = float(v)
			if kk not in self.upper:
				self.upper[kk] = fv
			else:
				self.upper[kk] = min(self.upper[kk], fv)
		return dict(self.upper)
	def get(self):
		return dict(self.upper)

