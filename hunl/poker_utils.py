from collections import Counter
import itertools

RANKS = "23456789TJQKA"
SUITS = "CDHS"

DECK = [r + s for r in RANKS for s in SUITS]

RANK_VALUES = {r: i + 2 for i, r in enumerate(RANKS)}
RANK_VALUES["A"] = 14
RVAL = {r: i + 2 for i, r in enumerate(RANKS)}


class Card:
	STR_RANKS = "23456789TJQKA"
	STR_SUITS = "shdc"
	INT_RANKS = range(13)
	PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
	CHAR_RANK_TO_INT_RANK = dict(zip(list(STR_RANKS), INT_RANKS))
	CHAR_SUIT_TO_INT_SUIT = {
		"s": 1, "h": 2, "d": 4, "c": 8,
		"S": 1, "H": 2, "D": 4, "C": 8,
		"\u2660": 1, "\u2764": 2, "\u2666": 4, "\u2663": 8,
	}
	INT_SUIT_TO_CHAR_SUIT = "xshxdxxxc"
	PRETTY_SUITS = {1: chr(9824), 2: chr(9829), 4: chr(9830), 8: chr(9827)}
	SUIT_COLORS = {2: "red", 4: "blue", 8: "green"}

	@staticmethod
	def new(string: str) -> int:
		rank_char = string[0]
		suit_char = string[1]
		rank_int = Card.CHAR_RANK_TO_INT_RANK[rank_char]
		suit_int = Card.CHAR_SUIT_TO_INT_SUIT[suit_char]
		rank_prime = Card.PRIMES[rank_int]
		bitrank = 1 << rank_int << 16
		suit = suit_int << 12
		rank = rank_int << 8
		return bitrank | suit | rank | rank_prime

	@staticmethod
	def get_rank_int(card_int: int) -> int:
		return (card_int >> 8) & 0xF

	@staticmethod
	def get_suit_int(card_int: int) -> int:
		return (card_int >> 12) & 0xF

	@staticmethod
	def prime_product_from_hand(card_ints) -> int:
		p = 1
		for c in card_ints:
			p *= (c & 0xFF)
		return p

	@staticmethod
	def prime_product_from_rankbits(rankbits: int) -> int:
		p = 1
		for i in Card.INT_RANKS:
			if rankbits & (1 << i):
				p *= Card.PRIMES[i]
		return p


class LookupTable:
	MAX_ROYAL_FLUSH = 1
	MAX_STRAIGHT_FLUSH = 10
	MAX_FOUR_OF_A_KIND = 166
	MAX_FULL_HOUSE = 322
	MAX_FLUSH = 1599
	MAX_STRAIGHT = 1609
	MAX_THREE_OF_A_KIND = 2467
	MAX_TWO_PAIR = 3325
	MAX_PAIR = 6185
	MAX_HIGH_CARD = 7462
	MAX_TO_RANK_CLASS = {
		MAX_ROYAL_FLUSH: 0,
		MAX_STRAIGHT_FLUSH: 1,
		MAX_FOUR_OF_A_KIND: 2,
		MAX_FULL_HOUSE: 3,
		MAX_FLUSH: 4,
		MAX_STRAIGHT: 5,
		MAX_THREE_OF_A_KIND: 6,
		MAX_TWO_PAIR: 7,
		MAX_PAIR: 8,
		MAX_HIGH_CARD: 9,
	}
	RANK_CLASS_TO_STRING = {
		0: "Royal Flush",
		1: "Straight Flush",
		2: "Four of a Kind",
		3: "Full House",
		4: "Flush",
		5: "Straight",
		6: "Three of a Kind",
		7: "Two Pair",
		8: "Pair",
		9: "High Card",
	}

	def __init__(self):
		self.flush_lookup = {}
		self.unsuited_lookup = {}
		self.flushes()
		self.multiples()

	def flushes(self):
		straight_flushes = [7936, 3968, 1984, 992, 496, 248, 124, 62, 31, 4111]
		flushes = []
		gen = self.get_lexographically_next_bit_sequence(int("0b11111", 2))
		i = 0
		total = 1277 + len(straight_flushes) - 1
		while i < total:
			f = next(gen)
			ok = True
			for sf in straight_flushes:
				if not f ^ sf:
					ok = False
			if ok:
				flushes.append(f)
			i += 1
		flushes.reverse()
		rank = 1
		for sf in straight_flushes:
			pp = Card.prime_product_from_rankbits(sf)
			self.flush_lookup[pp] = rank
			rank += 1
		rank = LookupTable.MAX_FULL_HOUSE + 1
		for f in flushes:
			pp = Card.prime_product_from_rankbits(f)
			self.flush_lookup[pp] = rank
			rank += 1
		self.straight_and_highcards(straight_flushes, flushes)

	def straight_and_highcards(self, straights, highcards):
		rank = LookupTable.MAX_FLUSH + 1
		for s in straights:
			pp = Card.prime_product_from_rankbits(s)
			self.unsuited_lookup[pp] = rank
			rank += 1
		rank = LookupTable.MAX_PAIR + 1
		for h in highcards:
			pp = Card.prime_product_from_rankbits(h)
			self.unsuited_lookup[pp] = rank
			rank += 1

	def multiples(self):
		back = list(range(12, -1, -1))
		rank = LookupTable.MAX_STRAIGHT_FLUSH + 1
		for i in back:
			kickers = back[:]
			kickers.remove(i)
			for k in kickers:
				p = Card.PRIMES[i] ** 4 * Card.PRIMES[k]
				self.unsuited_lookup[p] = rank
				rank += 1
		rank = LookupTable.MAX_FOUR_OF_A_KIND + 1
		for i in back:
			pairs = back[:]
			pairs.remove(i)
			for pr in pairs:
				p = Card.PRIMES[i] ** 3 * Card.PRIMES[pr] ** 2
				self.unsuited_lookup[p] = rank
				rank += 1
		rank = LookupTable.MAX_STRAIGHT + 1
		for r in back:
			kickers = back[:]
			kickers.remove(r)
			for c1, c2 in itertools.combinations(kickers, 2):
				p = Card.PRIMES[r] ** 3 * Card.PRIMES[c1] * Card.PRIMES[c2]
				self.unsuited_lookup[p] = rank
				rank += 1
		rank = LookupTable.MAX_THREE_OF_A_KIND + 1
		for p1, p2 in itertools.combinations(tuple(back), 2):
			kickers = back[:]
			kickers.remove(p1)
			kickers.remove(p2)
			for k in kickers:
				p = Card.PRIMES[p1] ** 2 * Card.PRIMES[p2] ** 2 * Card.PRIMES[k]
				self.unsuited_lookup[p] = rank
				rank += 1
		rank = LookupTable.MAX_TWO_PAIR + 1
		for pairrank in back:
			kickers = back[:]
			kickers.remove(pairrank)
			for k1, k2, k3 in itertools.combinations(tuple(kickers), 3):
				p = Card.PRIMES[pairrank] ** 2 * Card.PRIMES[k1] * Card.PRIMES[k2] * Card.PRIMES[k3]
				self.unsuited_lookup[p] = rank
				rank += 1

	def get_lexographically_next_bit_sequence(self, bits: int):
		t = int((bits | (bits - 1))) + 1
		nx = t | ((int(((t & -t) / (bits & -bits))) >> 1) - 1)
		yield nx
		while True:
			t = (nx | (nx - 1)) + 1
			nx = t | ((((t & -t) // (nx & -nx)) >> 1) - 1)
			yield nx


class Evaluator:
	def __init__(self):
		self.table = LookupTable()

	def _five(self, cards):
		if cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000:
			handOR = (cards[0] | cards[1] | cards[2] | cards[3] | cards[4]) >> 16
			prime = Card.prime_product_from_rankbits(handOR)
			return self.table.flush_lookup[prime]
		else:
			prime = Card.prime_product_from_hand(cards)
			return self.table.unsuited_lookup[prime]

	def _six(self, cards):
		m = LookupTable.MAX_HIGH_CARD
		for combo in itertools.combinations(cards, 5):
			sc = self._five(combo)
			if sc < m:
				m = sc
		return m

	def _seven(self, cards):
		m = LookupTable.MAX_HIGH_CARD
		for combo in itertools.combinations(cards, 5):
			sc = self._five(combo)
			if sc < m:
				m = sc
		return m

	def evaluate(self, hand, board):
		allc = hand + board
		n = 0
		for _ in allc:
			n += 1
		if n == 5:
			return self._five(allc)
		if n == 6:
			return self._six(allc)
		return self._seven(allc)


_eval = Evaluator()


def get_rank(card):
	return card[0]


def get_suit(card):
	return card[1]


def _ascii_upper(s):
	out = ""
	try:
		for ch in s:
			try:
				oc = ord(ch)
				if 97 <= oc <= 122:
					out = out + chr(oc - 32)
				else:
					out = out + ch
			except Exception:
				out = out + ch
	except Exception:
		t = str(s)
		for ch in t:
			try:
				oc = ord(ch)
				if 97 <= oc <= 122:
					out = out + chr(oc - 32)
				else:
					out = out + ch
			except Exception:
				out = out + ch
	return out


def _strip_spaces_ascii(s):
	ws = {9, 10, 11, 12, 13, 32}
	out = ""
	try:
		for ch in s:
			try:
				oc = ord(ch)
				if oc in ws:
					continue
			except Exception:
				pass
			out = out + ch
	except Exception:
		t = str(s)
		for ch in t:
			try:
				oc = ord(ch)
				if oc in ws:
					continue
			except Exception:
				pass
			out = out + ch
	return out


def _concat_sequence_raw(x):
	out = ""
	try:
		for t in x:
			if type(t) is str:
				out = out + t
			else:
				out = out + str(t)
	except Exception:
		out = str(x)
	return out


def _tokenize_ws(s):
	ws = {9, 10, 11, 12, 13, 32}
	out = []
	cur = ""
	try:
		it = iter(s)
	except Exception:
		it = iter(str(s))
	for ch in it:
		ok = True
		try:
			oc = ord(ch)
			if oc in ws:
				ok = False
		except Exception:
			ok = True
		if ok:
			cur = cur + ch
		else:
			if cur:
				out.append(cur)
				cur = ""
	if cur:
		out.append(cur)
	return out


def _to_str_card(c):
	if type(c) is str:
		raw = c
	elif (type(c) is list) or (type(c) is tuple):
		raw = _concat_sequence_raw(c)
	else:
		raw = str(c)
	s0 = _strip_spaces_ascii(raw)
	s = _ascii_upper(s0)
	if (s and s[0:1]) and (s[1:2]):
		r0 = s[0]
		t0 = s[1]
		if (r0 in RANKS) and (t0 in SUITS):
			return r0 + t0
		if (r0 in SUITS) and (t0 in RANKS):
			return t0 + r0
	if (s and s[0:1] == "1") and (s[1:2] == "0") and (s[2:3] in SUITS):
		return "T" + s[2]
	r = None
	t = None
	try:
		it = iter(s)
	except Exception:
		it = iter(str(s))
	i = 0
	buf = []
	for ch in it:
		buf.append(ch)
		i += 1
	n = i
	i = 0
	while i < n:
		ch = buf[i]
		if r is None:
			if (ch == "1") and (i + 1 < n) and (buf[i + 1] == "0"):
				r = "T"
				i += 1
			elif ch in RANKS:
				r = ch
		if (t is None) and (ch in SUITS):
			t = ch
		if (r is not None) and (t is not None):
			break
		i += 1
	if (r is not None) and (t is not None):
		return r + t
	return s[:2]


def _normalize_cards(card_iterable):
	if (type(card_iterable) is list) or (type(card_iterable) is tuple) or (type(card_iterable) is set):
		items = card_iterable
	elif type(card_iterable) is str:
		items = _tokenize_ws(card_iterable)
	else:
		items = [card_iterable]
	out = []
	for it in items:
		out.append(_to_str_card(it))
	return out


def _straight_high_from_ranks(rvals):
	u = sorted(set(rvals))
	if len(u) != 5:
		return None
	if u == [2, 3, 4, 5, 14]:
		return 5
	i = 0
	while i < 4:
		if (u[i + 1] - u[i]) != 1:
			return None
		i += 1
	return u[-1]


def _five_card_hand_key(hand5_strs):
	h = []
	for x in hand5_strs:
		h.append(_to_str_card(x))
	c_int = []
	for c in h:
		c_int.append(Card.new(c))
	r = _eval._five(c_int)
	return (10000 - int(r),)


def _best_five_from_seven_key(hole_or_five, board):
	cards7 = _normalize_cards(list(hole_or_five) + list(board))
	best_rank = None
	for comb in itertools.combinations(cards7, 5):
		r = _eval._five([Card.new(c) for c in comb])
		if (best_rank is None) or (r < best_rank):
			best_rank = r
	return (10000 - int(best_rank),)


def hand_rank(hole_or_five, board=None):
	if board is not None:
		return _best_five_from_seven_key(hole_or_five, board)
	else:
		hand5 = _normalize_cards(hole_or_five)
		return _five_card_hand_key(hand5)


def best_hand(cards):
	best = None
	best_r = None
	ls = _normalize_cards(cards)
	for comb in itertools.combinations(ls, 5):
		r = _eval._five([Card.new(c) for c in comb])
		if (best_r is None) or (r < best_r):
			best_r = r
			best = comb
	return best


def card_to_index(card):
	cs = _to_str_card(card)
	r = cs[0]
	s = cs[1]
	ri = RANKS.index(r)
	si = SUITS.index(s)
	return (ri * len(SUITS)) + si


def board_one_hot(board):
	vec = [0] * (len(RANKS) * len(SUITS))
	for c in board:
		k = card_to_index(c)
		if (0 <= k) and (k < len(vec)):
			vec[k] = 1
	return vec


def evaluate_7card(hole, board):
	return hand_rank(hole, board)

