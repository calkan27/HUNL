"""
I am a lightweight container for hand→bucket mappings and conversions between hand-space
and bucket-space distributions. I normalize probabilities, enforce single-bucket
membership per hand, and expose helpers to move back and forth between spaces.

Key class: BucketedRange. Key methods: buckets — list available bucket ids;
hand_to_bucket/bucket_to_hands — mapping queries; from_hand_probs — compress a hand PMF
into bucket probabilities; to_hand_probs — expand a bucket distribution evenly across
member hands.

Inputs: mapping {bucket_id → set(hands)} and optional explicit bucket count; hand
strings are expected in canonical "R1S1 R2S2" form. Outputs: normalized bucket vectors
or per-hand probability maps.

Internal dependencies: none. External dependencies: none.

Invariants: one hand maps to at most one bucket; output distributions are normalized
when mass is positive; expansion divides bucket mass equally over its member hands.
Performance: linear in number of hands; deterministic ordering for reproducible tests.
"""

class BucketedRange:
	def __init__(self, bucket_mapping, num_buckets=None):
		_bm = dict(bucket_mapping)
		mapping = {}
		for k, v in _bm.items():
			mapping[int(k)] = set(v)
		self.mapping = mapping

		if num_buckets is not None:
			self.num_buckets = int(num_buckets)
		else:
			self.num_buckets = len(self.mapping)

		self._hand_to_bucket = {}

		for cid, hs in self.mapping.items():
			for h in hs:
				self._hand_to_bucket[h] = int(cid)


	def buckets(self):
		return list(range(self.num_buckets))

	def hand_to_bucket(self, hand):
		key = hand if isinstance(hand, str) else " ".join(list(hand))
		return int(self._hand_to_bucket.get(key, -1))

	def bucket_to_hands(self, bucket_id):
		return sorted(list(self.mapping.get(int(bucket_id), set())))

	def from_hand_probs(self, hand_probs):
		vec = [0.0] * self.num_buckets
		s = 0.0
		for h, p in dict(hand_probs).items():
			key = h if isinstance(h, str) else " ".join(list(h))
			cid = self._hand_to_bucket.get(key, None)
			if cid is None:
				continue
			vec[int(cid)] += float(p)
			s += float(p)
		if s > 0.0:
			i = 0
			while i < self.num_buckets:
				vec[i] = vec[i] / s
				i += 1
		return vec

	def to_hand_probs(self, bucket_probs):
		out = {}
		total = 0.0
		for cid, p in enumerate(list(bucket_probs)):
			hands = sorted(list(self.mapping.get(int(cid), set())))
			if not hands:
				continue
			w = float(p) / float(len(hands))
			for h in hands:
				out[h] = w
				total += w
		if total > 0.0:
			for k in list(out.keys()):
				out[k] = out[k] / total
		return out

