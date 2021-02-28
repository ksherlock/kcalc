#
# python 3

import readline
import re
from functools import reduce

# WS = re.compile("[ \t]*")

from number import Number, unary_op, binary_op, logical_or, logical_and, logical_xor, coerce
from number import default_type, int32_t, uint32_t, int16_t, uint16_t


def _binary(fn):
	def inner(a,b):
		return binary_op(a,b, fn)
	return inner


def _unary(fn):
	def inner(a):
		return unary_op(a, fn)
	return inner


def c_division(a,b):
	# C++11 division rounds towards 0.
	# this is inconsistent w/ python division.
	if (a < 0) == (b < 0): return a // b
	return -(abs(a) // abs(b))

def c_modulo(a,b):
	# (a/b)*b + a%b == a
	if (a < 0) == (b < 0): return a % b
	return -(abs(a) % abs(b))

def pascal_modulo(a,b):
	# The mod operator returns the remainder resulting from the division of two integer quantities. 
	# The mod operator for i mod j is defined as the smallest positive number that can result from the
 	# expression (i - (k * j)), where k is also an integer.

 	# https://wiki.freepascal.org/Mod

	if (a < 0) == (b < 0): return a % b
	return abs(a) % abs(b)

def orca_shift(a,b):
	# -b is a shift right
	b = coerce(b, int32_t)
	if b < 0: return a >> b
	return a << b 


class Parser(object):
	"""docstring for Parser"""
	default_type = None

	def __init__(self):
		pass

	def evaluate(self, s, env={}):
		self._tokens = self._tokenize(s, env)
		x = self._expr()
		if self._tokens: raise Exception("Syntax error")
		x.value() # raise exception
		return x


	def _tokenize(self, s, env):
		offset = 0
		tokens = []
		s = s.strip()
		l = len(s)
		while offset < l:
			m = self.RE.match(s, offset)

			if not m: raise Exception("Syntax Error")
			offset = m.end()
			value = None

			if m["op"]: value = m["op"]

			elif m["bin"]: value = Number(int(m["bin"], 2))

			elif m["hex"]: value = Number(int(m["hex"], 16))

			elif m["dec"]: value = Number(int(m["dec"], 10))

			elif m["oct"]: value = Number(int(m["oct"], 8))

			elif m["id"]: value = env[m["id"]]

			elif m["cc"]:
				xx = [ord(x) for x in m["cc"]]
				xx.reverse() # make little endian
				value = Number(reduce(lambda x,y: (x << 8) + y, xx))
			else: raise Exception("Missing type...")

			tokens.append(value)

		tokens.reverse()
		return tokens


	def _token(self):
		if not self._tokens: return None
		return self._tokens.pop()

	def _term(self):
		tk = self._token()
		if tk == '(': return self._expr(')')
		#if type(tk) == int: return tk
		if type(tk) == Number: return tk
		if tk in self.UNARY:
			prec, fn = self.UNARY[tk]
			t = self._term()
			if t == None: raise Exception("Syntax error: expected terminal")
			return fn(t)
		raise Exception("Syntax error: expected terminal")
		

	def _expr(self, end = None):

		end_set = (end, None)
		operands = []
		operators = []

		while True:
			t = self._term()
			if t == None: break
			operands.append(t)
			tk = self._token()
			if tk in end_set: break

			if not tk in self.BINARY: raise Exception("Bad operator: {}".format(tk))
			op = self.BINARY[tk]

			while operators and operators[-1][0] <= op[0]:
				a = operands.pop()
				b = operands.pop()
				_, fn = operators.pop()
				operands.append(fn(b, a))
			operators.append(op)

		#
		while operators:

			a = operands.pop()
			b = operands.pop()
			_, fn = operators.pop()
			operands.append(fn(b, a))

		if len(operands) != 1: raise Exception("Syntax Error")
		return operands[0]


class CParser(Parser):

	default_type = int32_t

	RE = re.compile(r"""
			(?:[ \t]*)
			(?:
				  0x(?P<hex>[A-Fa-f0-9]+)
				| 0b(?P<bin>[01]+)
				| (?P<oct>0[0-7]*)
				| (?P<dec>[1-9][0-9]*)
				| (?P<id>[_A-Za-z][_A-Za-z0-9]*)
				| '(?P<cc>[^'\x00-\x1fx7f]{1,4})'
				| (?P<op><<|>>|<=|>=|==|!=|&&|\|\||[-+=<>~*!~/%^&|()<>])
			)
		""", re.X)

	BINARY = {

		'*': (3, _binary(lambda x,y: x*y)),
		'/': (3, _binary(lambda x,y: x//y)),
		'%': (3, _binary(lambda x,y: x%y)),
		'+': (4, _binary(lambda x,y: x+y)),
		'-': (4, _binary(lambda x,y: x-y)),
		'<': (6, _binary(lambda x,y: x<y)),
		'>': (6, _binary(lambda x,y: x>y)),
		'&': (8, _binary(lambda x,y: x&y)),
		'^': (9, _binary(lambda x,y: x^y)),
		'|': (10, _binary(lambda x,y: x|y)),

		'&&': (11, logical_and ),
		'||': (12, logical_or),

		'<<': (5, _binary(lambda x,y: x<<y)),
		'>>': (5, _binary(lambda x,y: x>>y)),
		'<=': (6, _binary(lambda x,y: x<=y)),
		'>=': (6, _binary(lambda x,y: x>=y)),
		'==': (7, _binary(lambda x,y: x==y)),
		'!=': (7, _binary(lambda x,y: x!=y)),
	}

	UNARY = {
		'+': (2, _unary(lambda x: +x)),
		'-': (2, _unary(lambda x: -x)),
		'~': (2, _unary(lambda x: ~x)),
		'!': (2, _unary(lambda x: not x)),
	}

	def __init__(self):
		super(CParser, self).__init__()


class PascalParser(Parser):

	default_type = int32_t

	RE = re.compile(r"""
			(?:[ \t]*)
			(?:
				  \$(?P<hex>[A-Fa-f0-9]+)
				| %(?P<bin>[01]+)
				| (?P<dec>[0-9]+)
				| '(?P<cc>[^'\x00-\x1fx7f]{1,4})'
				| (?P<op>div|mod|and|or|not|xor|<<|>>|<=|>=|<>|\*\*|:=|[-+*/=<>&|!~()])
				| (?P<id>[_A-Za-z][_A-Za-z0-9]*)
			)
		""", re.X | re.I)

	BINARY = {
		'**': (2, _binary(lambda x,y: x**y)),

		'*': (3, _binary(lambda x,y: x*y)),
		'/': (3, _binary(lambda x,y: x//y)),
		'&': (3, _binary(lambda x,y: x&y)),
		'<<': (3, _binary(lambda x,y: x<<y)),
		'>>': (3, _binary(lambda x,y: x>>y)),
		'div': (3, _binary(lambda x,y: x//y)),
		'mod': (3, _binary(lambda x,y: x%y)),
		'and': (3, logical_and),

		'+': (4, _binary(lambda x,y: x+y)),
		'-': (4, _binary(lambda x,y: x-y)),
		'|': (4, _binary(lambda x,y: x|y)),
		'!': (4, _binary(lambda x,y: x^y)),
		'xor': (4, _binary(lambda x,y: x^y)),
		'or': (4, logical_or),

		'=': (5, _binary(lambda x,y: x==y)),
		'<': (5, _binary(lambda x,y: x<y)),
		'>': (5, _binary(lambda x,y: x>y)),
		'<=': (5, _binary(lambda x,y: x<=y)),
		'>=': (5, _binary(lambda x,y: x>=y)),
		'<>': (5, _binary(lambda x,y: x!=y)),
	}

	UNARY = {
		'+': (2, _unary(lambda x: +x)),
		'-': (2, _unary(lambda x: -x)),
		'~': (2, _unary(lambda x: ~x)),
		'not': (2, _unary(lambda x: not x)),
	}

	def __init__(self):
		super(PascalParser, self).__init__()


class MerlinParser(Parser):

	default_type = uint32_t

	# should also have "x" for high-ascii char constant.
	RE = re.compile(r"""
			(?:[ \t]*)
			(?:
				  \$(?P<hex>[A-Fa-f0-9]+)
				| %(?P<bin>[01]+)
				| (?P<dec>[0-9]+)
				| '(?P<cc>[^'\x00-\x1fx7f]{1,4})'
				| (?P<op>[-+*/!.&])
				| (?P<id>[_A-Za-z][_A-Za-z0-9]*)
			)
		""", re.X | re.I)

	BINARY = {
		'*': (1, _binary(lambda x,y: x*y)),
		'/': (1, _binary(lambda x,y: x//y)),
		'+': (1, _binary(lambda x,y: x+y)),
		'-': (1, _binary(lambda x,y: x-y)),
		'&': (1, _binary(lambda x,y: x&y)),
		'.': (1, _binary(lambda x,y: x|y)),
		'!': (1, _binary(lambda x,y: x^y)),
	}

	UNARY = {
		'+': (1, _unary(lambda x: +x)),
		'-': (1, _unary(lambda x: -x)),	
	}

	def __init__(self):
		super(MerlinParser, self).__init__()


class OrcaParser(Parser):

	default_type = uint32_t

	RE = re.compile(r"""
			(?:[ \t]*)
			(?:
				  \$(?P<hex>[A-Fa-f0-9]+)
				| %(?P<bin>[01]+)
				| (?P<dec>[0-9]+)
				| @(?P<oct>[0-7]+)
				| '(?P<cc>[^'\x00-\x1fx7f]{1,4})'
				| (?P<op>\.NOT\.|\.OR\.|\.EOR\.|\.AND\.|<=|>=|<>|[-+*/|=<>()])
				| (?P<id>[_A-Za-z][_A-Za-z0-9]*)
			)
		""", re.X | re.I)

	BINARY = {

		'*': (3, _binary(lambda x,y: x*y)),
		'/': (3, _binary(lambda x,y: x//y)),
		'|': (3, orca_shift),
		'!': (3, orca_shift),

		'+': (4, _binary(lambda x,y: x+y)),
		'-': (4, _binary(lambda x,y: x-y)),
		'.OR.': (4, logical_or),
		'.EOR.': (4, logical_xor),
		'.AND.': (4, logical_and),

		'=': (5, _binary(lambda x,y: x==y)),
		'<': (5, _binary(lambda x,y: x<y)),
		'>': (5, _binary(lambda x,y: x>y)),
		'<=': (5, _binary(lambda x,y: x<=y)),
		'>=': (5, _binary(lambda x,y: x>=y)),
		'<>': (5, _binary(lambda x,y: x!=y)),
	}

	UNARY = {
		'+': (2, _unary(lambda x: +x)),
		'-': (2, _unary(lambda x: -x)),
		'~': (2, _unary(lambda x: ~x)),
		'.NOT.': (2, _unary(lambda x: not x)),
	}

	def __init__(self):
		super(PascalParser, self).__init__()


def to_b(v):
	rv = ""
	for i in range(0,32):
		rv += "01"[v & 0x01]
		v >>= 1
	return rv[::-1]

def to_cc(v):
	# little endian
	if not v: return ""
	b = []
	while v:
		b.append(v & 0xff)
		v >>= 8
	if not all([x>=0x20 and x<0x7f for x in b]): return ""

	return "'" + "".join( [chr(x) for x in b] ) + "'"


def display(num):
	# 0xffff 0b1111 'cccc'
	value = num.value()
	uvalue = num.unsigned_value()
	print("0x{:08x}  0b{:032b} {}".format(uvalue, uvalue, to_cc(uvalue)))

	for x in num.alternates() :
		print("{:24}".format(str(x)), end="")

	#if num.is_signed():
	#	print("(int32_t){:<10}  (uint32_t){:<10}".format(value, uvalue))
	#else:
	#	print("(uint32_t){:<10} (int32_t){:<10}".format(uvalue, value))

	#
	#print("{:10}".format(value))#, end="")
	#
	#tmp = num.alternates()
	#for n in tmp:
	#	print(str(n), end=" ")

	print()


if __name__ == '__main__':
	default_type = int32_t
	p = CParser()
	env = {}
	env["_"] = 0

	while True:
		s = ""
		try:
			s = input("] ")
			s.strip()
			if not s: continue
			x = p.evaluate(s, env)
			env["_"]=x
			display(x)
		except EOFError as ex:
			print()
			exit(0)
		except KeyboardInterrupt as ex:
			print()
			exit(0)

		except KeyError as ex:
			print("Unbound variable: {}".format(ex.args[0]))

		except Exception as e:
			print(e)

