from poker_utils import board_one_hot

class Weird:
	def __str__(self):
		# deliberately messy but valid; should normalize to "TH"
		return " h 10 "

def test_no_recursion_in_board_encoding():
	v = board_one_hot(["JH", ("Q","s"), Weird()])
	assert isinstance(v, list) and len(v) == 52
	# three distinct cards => exactly three 1s
	assert sum(v) == 3

