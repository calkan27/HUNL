"""
I compute per-cluster river EVs using either exact pairwise enumeration or a
strength-bucketing approximation. I filter impossible hands against the board,
optionally sample per-cluster candidate sets, and aggregate payoffs into vectors aligned
with action indices so upstream code can reuse them without translation.

Key class: RiverEndgame. Key methods: compute_cluster_cfvs — main entry; _ev_no_bucket —
exact pairwise EV accumulation with optional pot scaling; river_endgame_bucket_probs_for
— histogram over strength buckets; river_endgame_pay — payoff matrix kernel;
river_endgame_norm_vec — vector normalization. Important helpers:
_filtered_hands_for_cluster, _cluster_distribution, _bucketize.

Inputs: clusters, GameNode (to read board and pot), player id, and three helpers:
wins_fn, best_hand_fn, hand_rank_fn. Outputs: dict {cluster_id: [ev, ev, ev, ev]}
compatible with ActionType indices.

Dependencies: stdlib; upstream provides poker evaluation helpers. Invariants: EV
symmetry between players; bucketed and exact paths return consistent signs; ranges are
normalized before mixing. Performance: sampling and bucketing control complexity for
large clusters.
"""

from hunl.constants import SEED_RIVER
import random


class RiverEndgame:
	def __init__(self, num_buckets=None, max_sample_per_cluster=None, seed=SEED_RIVER):
		self.num_buckets = num_buckets
		self.max_sample_per_cluster = max_sample_per_cluster
		self.seed = int(seed)

	def _filter_hands(self, hands, board_set):
		out = []
		for h in hands:
			if isinstance(h, str):
				c1, c2 = h.split()
			else:
				c1, c2 = list(h)
			ok_pair = (c1 != c2)
			ok_board = (c1 not in board_set) and (c2 not in board_set)
			if ok_pair:
				if ok_board:
					if isinstance(h, str):
						out.append(h)
					else:
						out.append(f"{c1} {c2}")
		return out

	def _sample(self, items, k, key):
		if (k is None) or (k <= 0) or (len(items) <= k):
			return list(items)
		else:
			rng = random.Random((self.seed * 1315423911) ^ int(key))
			return rng.sample(list(items), int(k))

	def _strength_key(self, hand_rank_tuple):
		return hand_rank_tuple

	def _bucketize(self, strength_list):
		if (self.num_buckets is None) or (self.num_buckets <= 0):
			return None, None
		else:
			uniq = sorted(set(strength_list))
			if len(uniq) <= int(self.num_buckets):
				bmap = {}
				i = 0
				while i < len(uniq):
					bmap[uniq[i]] = i
					i += 1
				return bmap, len(uniq)
			else:
				idx_map = {}
				n = len(uniq)
				i = 0
				while i < n:
					b = int(i * int(self.num_buckets) / n)
					if b >= int(self.num_buckets):
						b = int(self.num_buckets) - 1
					idx_map[uniq[i]] = b
					i += 1
				return idx_map, int(self.num_buckets)

	def _cluster_distribution(self, hands, board, best_hand_fn, hand_rank_fn):
		strengths = []
		board_n = [str(c)[0].upper() + str(c)[1].lower() for c in list(board)]
		for h in hands:
			c1, c2 = h.split()
			h1 = str(c1)[0].upper() + str(c1)[1].lower()
			h2 = str(c2)[0].upper() + str(c2)[1].lower()
			hand = [h1, h2]
			r = hand_rank_fn(best_hand_fn(hand + board_n))
			strengths.append(self._strength_key(r))
		return strengths

	def _pairwise_util_p(self, res, pot_size=None, my_bet=None, opp_bet=None, resolved_pot=None):
		if (my_bet is not None) or (opp_bet is not None):
			mb = float(my_bet or 0.0)
			ob = float(opp_bet or 0.0)
			if res > 0:
				return ob
			else:
				if res < 0:
					return -mb
				else:
					return 0.5 * (ob - mb)
		else:
			if resolved_pot is not None:
				P = float(resolved_pot)
			else:
				if pot_size is not None:
					P = float(pot_size)
				else:
					P = 1.0
			if res > 0:
				return P
			else:
				if res < 0:
					return -P
				else:
					return 0.0

	def _expected_utility_buckets_both(self, my_bucket_probs, opp_bucket_probs, B, resolved_pot=None, my_bet=None, opp_bet=None):
		p = self.river_endgame_norm_vec(list(my_bucket_probs))
		q = self.river_endgame_norm_vec(list(opp_bucket_probs))
		B = int(B)
		bet_mode = (my_bet is not None) or (opp_bet is not None)

		if bet_mode:
			mb = float(my_bet or 0.0)
			ob = float(opp_bet or 0.0)
			P = 0.0
		else:
			mb = 0.0
			ob = 0.0
			if resolved_pot is not None:
				P = float(resolved_pot)
			else:
				P = 1.0

		if not bet_mode:
			s = 0.0
			i = 0
			while i < B:
				if float(p[i]) != 0.0:
					j = 0
					while j < B:
						if float(q[j]) != 0.0:
							if i > j:
								s += float(p[i]) * float(q[j])
							else:
								if i < j:
									s -= float(p[i]) * float(q[j])
						j += 1
				i += 1
			ev_p = float(P) * float(s)
			return float(ev_p), float(-ev_p)

		ev_p = 0.0
		i = 0
		while i < B:
			if float(p[i]) != 0.0:
				row = 0.0
				j = 0
				while j < B:
					if float(q[j]) != 0.0:
						val = self.river_endgame_pay(i, j, True, mb, ob, 0.0)
						row += float(q[j]) * float(val)
					j += 1
				ev_p += float(p[i]) * row
			i += 1

		ev_o = 0.0
		j = 0
		while j < B:
			if float(q[j]) != 0.0:
				col = 0.0
				i = 0
				while i < B:
					if float(p[i]) != 0.0:
						val2 = self.river_endgame_pay(j, i, True, ob, mb, 0.0)
						col += float(p[i]) * float(val2)
					i += 1
				ev_o += float(q[j]) * col
			j += 1

		return float(ev_p), float(ev_o)

	def _expected_utility_pairwise(self, my_hand, opp_hand, board, wins_fn, pot_size=None, my_bet=None, opp_bet=None, resolved_pot=None):
		res = wins_fn(my_hand, opp_hand, board)
		up = self._pairwise_util_p(
		 res,
		 pot_size=pot_size,
		 my_bet=my_bet,
		 opp_bet=opp_bet,
		 resolved_pot=resolved_pot,
		)
		return up, -up

	def _normalize_range(self, r):
		out = {}
		items = dict(r).items()
		for k, v in items:
			out[int(k)] = float(v)
		s = 0.0
		for v in out.values():
			s += float(v)
		if s > 0.0:
			for k in list(out.keys()):
				out[k] = out[k] / s
		else:
			for k in list(out.keys()):
				out[k] = 0.0
		return out


	def _filtered_hands_for_cluster(self, clusters, cid, board_set_upper, sample_ok=True):
		out = []
		hset = clusters.get(int(cid), [])
		for h in hset:
			if isinstance(h, str):
				a, b = h.split()
			else:
				a, b = list(h)
			if a == b:
				continue
			in_board = (str(a).upper() in board_set_upper) or (str(b).upper() in board_set_upper)
			if in_board:
				continue
			if isinstance(h, str):
				out.append(h)
			else:
				out.append(f"{a} {b}")
		if sample_ok:
			if isinstance(self.max_sample_per_cluster, int):
				if self.max_sample_per_cluster > 0:
					out = self._sample(out, self.max_sample_per_cluster, key=cid)
		return out

	def _ev_no_bucket(self, clusters, board_list_norm, board_set_upper, my_range, opp_range, wins_fn, resolved_pot):
		ev_p_by_cluster = {}
		ev_o_by_cluster = {}

		for cid in my_range.keys():
			my_hands = self._filtered_hands_for_cluster(clusters, cid, board_set_upper, sample_ok=False)

			if not my_hands:
				ev_p_by_cluster[int(cid)] = 0.0
			else:
				ev_total_p = 0.0
				my_w = 1.0 / float(len(my_hands))

				for my_h in my_hands:
					c1, c2 = my_h.split()
					m1 = str(c1)[0].upper() + str(c1)[1].lower()
					m2 = str(c2)[0].upper() + str(c2)[1].lower()
					my_cards = [m1, m2]

					for oid, oprob in opp_range.items():
						if float(oprob) <= 0.0:
							continue
						else:
							opp_hands = self._filtered_hands_for_cluster(clusters, oid, board_set_upper, sample_ok=False)

							if not opp_hands:
								continue
							else:
								if len(opp_hands) > 0:
									opp_w = float(oprob) / float(len(opp_hands))
								else:
									opp_w = 0.0

								for o_h in opp_hands:
									d1, d2 = o_h.split()
									o1 = str(d1)[0].upper() + str(d1)[1].lower()
									o2 = str(d2)[0].upper() + str(d2)[1].lower()
									opp_cards = [o1, o2]

									res = wins_fn(my_cards, opp_cards, list(board_list_norm))
									up = self._pairwise_util_p(res, resolved_pot=resolved_pot)
									ev_total_p += my_w * opp_w * up

									if int(oid) in ev_o_by_cluster:
										pass
									else:
										ev_o_by_cluster[int(oid)] = 0.0

									ev_o_by_cluster[int(oid)] += my_w * opp_w * (-up)

				ev_p_by_cluster[int(cid)] = ev_total_p

		for k in list(opp_range.keys()):
			if int(k) in ev_o_by_cluster:
				pass
			else:
				ev_o_by_cluster[int(k)] = 0.0

		out = {}

		if resolved_pot is not None:
			P = float(resolved_pot)
		else:
			P = 0.0

		if P > 0.0:
			sc = 1.0 / P
		else:
			sc = 1.0

		for cid, v in ev_p_by_cluster.items():
			out[int(cid)] = [float(v) * sc, float(v) * sc, float(v) * sc, float(v) * sc]

		return out, ev_o_by_cluster

	def compute_cluster_cfvs(self, clusters, node, player, wins_fn, best_hand_fn, hand_rank_fn):
		ps = node.public_state
		board = list(getattr(ps, "board_cards", []))
		board_list_norm = [str(c)[0].upper() + str(c)[1].lower() for c in board]
		board_set_upper = set([str(c).upper() for c in board])

		r_my_raw = dict(node.player_ranges[int(player)])
		r_opp_raw = dict(node.player_ranges[(int(player) + 1) % 2])

		r_my = self._normalize_range(r_my_raw)
		r_opp = self._normalize_range(r_opp_raw)

		if (self.num_buckets is None) or (int(self.num_buckets) <= 0):
			ev_p_by_cluster, _ = self._ev_no_bucket(
			 clusters,
			 board_list_norm,
			 board_set_upper,
			 r_my,
			 r_opp,
			 wins_fn,
			 resolved_pot=float(getattr(ps, "pot_size", 0.0)),
			)
			out0 = {}
			for cid, v in ev_p_by_cluster.items():
				out0[int(cid)] = list(v)
			return out0

		strengths_all = []
		per_cluster_strengths = {}

		keys_union = set(list(r_my.keys()) + list(r_opp.keys()))
		for cid in keys_union:
			cid_i = int(cid)
			hands = self._filtered_hands_for_cluster(clusters, cid_i, board_set_upper)
			if not hands:
				per_cluster_strengths[cid_i] = []
			else:
				slist = self._cluster_distribution(hands, board_list_norm, best_hand_fn, hand_rank_fn)
				per_cluster_strengths[cid_i] = slist
				strengths_all.extend(slist)

		strengths_source = []
		if strengths_all:
			strengths_source = strengths_all
		else:
			strengths_source = []

		bmap, B = self._bucketize(strengths_source)

		if bmap is None:
			ev_p_by_cluster, _ = self._ev_no_bucket(
			 clusters,
			 board_list_norm,
			 board_set_upper,
			 r_my,
			 r_opp,
			 wins_fn,
			 resolved_pot=float(getattr(ps, "pot_size", 0.0)),
			)
			out1 = {}
			for cid, v in ev_p_by_cluster.items():
				out1[int(cid)] = list(v)
			return out1

		opp_mix_bucket = [0.0] * int(B)
		for ocid, w in r_opp.items():
			q = self.river_endgame_bucket_probs_for(
			 per_cluster_strengths=per_cluster_strengths,
			 B=int(B),
			 bmap=bmap,
			 strength_key_fn=self._strength_key,
			 cid=int(ocid),
			)
			j = 0
			while j < int(B):
				opp_mix_bucket[j] += float(w) * q[j]
				j += 1

		sq = sum(opp_mix_bucket)
		if sq:
			pass
		else:
			sq = 0.0

		if sq > 0.0:
			j2 = 0
			while j2 < int(B):
				opp_mix_bucket[j2] = opp_mix_bucket[j2] / sq
				j2 += 1

		out = {}
		for cid, mass in r_my.items():
			pv = self.river_endgame_bucket_probs_for(
			 per_cluster_strengths=per_cluster_strengths,
			 B=int(B),
			 bmap=bmap,
			 strength_key_fn=self._strength_key,
			 cid=int(cid),
			)
			ev_p, _ = self._expected_utility_buckets_both(
			 pv,
			 opp_mix_bucket,
			 int(B),
			 resolved_pot=float(getattr(ps, "pot_size", 0.0)),
			 my_bet=None,
			 opp_bet=None,
			)
			P = float(getattr(ps, "pot_size", 0.0))

			if P > 0.0:
				sc = 1.0 / P
			else:
				sc = 1.0

			out[int(cid)] = [float(ev_p) * sc, float(ev_p) * sc, float(ev_p) * sc, float(ev_p) * sc]

		return out

	@staticmethod
	def river_endgame_norm_vec(v):
		out = []
		s = 0.0
		i = 0
		while i < len(v):
			val = float(v[i])
			out.append(val)
			s += val
			i += 1
		if s > 0.0:
			j = 0
			while j < len(out):
				out[j] = out[j] / s
				j += 1
		else:
			k = 0
			while k < len(out):
				out[k] = 0.0
				k += 1
		return out

	@staticmethod
	def river_endgame_pay(i, j, bet_mode, mb, ob, P):
		if bet_mode:
			if i > j:
				return float(ob)
			else:
				if i < j:
					return float(-mb)
				else:
					return 0.5 * (float(ob) - float(mb))
		else:
			if i > j:
				return float(P)
			else:
				if i < j:
					return float(-P)
				else:
					return 0.0

	@staticmethod
	def river_endgame_bucket_probs_for(per_cluster_strengths, B, bmap, strength_key_fn, cid):
		slist = per_cluster_strengths.get(int(cid), [])
		if not slist:
			return [0.0] * int(B)
		vec = [0.0] * int(B)
		i = 0
		while i < len(slist):
			s = slist[i]
			bi = int(bmap.get(strength_key_fn(s), 0))
			if 0 <= bi < int(B):
				vec[bi] += 1.0
			i += 1
		t = 0.0
		j = 0
		while j < int(B):
			t += vec[j]
			j += 1
		if t > 0.0:
			k = 0
			while k < int(B):
				vec[k] = vec[k] / t
				k += 1
		return vec

