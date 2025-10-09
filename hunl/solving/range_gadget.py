"""
Monotone, per-cluster upper-bound tracker used at depth limits and across re-solves. The
gadget stores an upper vector over opponent counterfactual values (CFVs) and exposes
begin/update/get methods that implement the classical follow/terminate carry-forward
pattern.

Key class: RangeGadget. Key methods: begin (initialize from a prior upper map), update
(componentwise max with newly observed uppers), get (snapshot as a plain dict).

Inputs: dictionaries mapping cluster id to upper CFV estimates, typically from a
re-solve at a public-state boundary. Outputs: a stable dict of upper bounds,
monotonically non-decreasing per cluster across updates.

Invariants: keys are coerced to ints, values to floats; updates are idempotent when
proposed values do not exceed the stored bound; begin preserves any prior state when
given empty input. Performance: O(K) memory and updates, designed to be negligible
compared to solving cost.
"""




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
				if self.upper[kk] >= fv:
					self.upper[kk] = float(self.upper[kk])
				else:
					self.upper[kk] = fv

		return dict(self.upper)

	def get(self):
		return dict(self.upper)

