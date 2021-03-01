#
# python 3

import readline
import re
from functools import reduce

# WS = re.compile("[ \t]*")

from number import Number, unary_op, binary_op, logical_or, logical_and, logical_xor
from type import *

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
	b = int32_t.cast(b)
	if b < 0: return a >> abs(b)
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

			if not m:
				print(s)
				print(" " * offset, "^", sep="")
				raise Exception("Syntax Error")
			offset = m.end()

			value = self._convert_token(m, env)

			tokens.append(value)

		tokens.reverse()
		return tokens


	def _convert_token(self, m, env):

		data = m.groupdict()

		x = data.get("op", None);
		if x: return x.lower()

		x = data.get("dec", None);
		if x: return Number(int(x, 10), self.default_type)
		x = data.get("hex", None);
		if x: return Number(int(x, 16), self.default_type)
		x = data.get("bin", None);
		if x: return Number(int(x, 2), self.default_type)
		x = data.get("oct", None);
		if x: return Number(int(x, 8), self.default_type)

		x = data.get("id", None);
		if x: return env[m["id"]]

		x = data.get("cc", None);
		if x:
			xx = [ord(y) for y in x]
			xx.reverse() # make little endian
			return Number(reduce(lambda x,y: (x << 8) + y, xx), self.default_type)
		raise Exception("Missing type...")



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
			if t == None: raise Exception("Syntax error: expected terminal, found EOF")
			return fn(t)
		if tk == None: tk = "EOF"
		raise Exception("Syntax error: expected terminal, found {}".format(tk))
		

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

	name = 'C'
	default_type = int32_t

	RE = re.compile(r"""
			(?:[ \t]*)
			(?:
			 	(
			 		(
			 		  0x(?P<hex>[A-Fa-f0-9]+)
					| 0b(?P<bin>[01]+)
					| (?P<oct>0[0-7]*)
					| (?P<dec>[1-9][0-9]*)
					)
					(?P<suffix>[uUlL]*)
				)
				| (?P<id>[_A-Za-z][_A-Za-z0-9]*)
				| '(?P<cc>[^'\x00-\x1fx7f]{1,4})'
				| (?P<op><<|>>|<=|>=|==|!=|&&|\|\||[-+=<>~*!~/%^&|()<>])
			)
		""", re.X)

	BINARY = {

		'*': (3, _binary(lambda x,y: x*y)),
		'/': (3, _binary(c_division)),
		'%': (3, _binary(c_modulo)),
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
	SUFFIX = {
		'u': None,
		'ul' : uint32_t,
		'lu' : uint32_t,
		'ull' : uint64_t,
		'llu' : uint64_t,
		'l': int32_t,
		'll': int64_t,
	}

	def _convert_token(self, m, env):

		if m["op"]: return m["op"]

		if m["cc"]:
			xx = [ord(x) for x in m["cc"]]
			xx.reverse() # make little endian
			return Number(reduce(lambda x,y: (x << 8) + y, xx))

		if m["id"]: return env[m["id"]]

		tp = self.default_type
		suffix = m["suffix"]
		if suffix:
			suffix = suffix.lower()
			if suffix not in self.SUFFIX:
				raise Exception("Invalid suffix: {}".format(suffix))
			tp = self.SUFFIX[suffix]
			if suffix == 'u':
				tp = self.default_type.make_unsigned()

		# integer literal types:
		# base 10 is always signed (unless explicitly unsigned)
		# other bases will covert to unsigned
		# eg, 0x7fff is signed int, 0xffff is unsigned int
		value = None

		base = 0
		if m["dec"]:   base = 10 ; value = int(m["dec"], 10)
		elif m["hex"]: base = 16 ; value = int(m["hex"], 16)
		elif m["bin"]: base = 2  ; value = int(m["bin"], 2)
		elif m["oct"]: base = 8  ; value = int(m["oct"], 8)
		else: raise Exception("Missing type... {}")

		signed = tp.is_signed()
		unsigned = tp.is_unsigned() or base != 10

		tp = Type.type_that_fits(
			value,
			base = tp,
			signed = signed,
			unsigned = unsigned
		)
		return Number(value, tp)



	def __init__(self):
		super(CParser, self).__init__()


class PascalParser(Parser):

	name = 'Pascal'
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
		'not': (2, _unary(lambda x: int(not x))),
	}

	def __init__(self):
		super(PascalParser, self).__init__()


class MerlinParser(Parser):

	name = 'Merlin'
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

	name = 'ORCA/M'
	default_type = uint32_t

	RE = re.compile(r"""
			(?:[ \t]*)
			(?:
				  \$(?P<hex>[A-Fa-f0-9]+)
				| %(?P<bin>[01]+)
				| (?P<dec>[0-9]+)
				| @(?P<oct>[0-7]+)
				| '(?P<cc>[^'\x00-\x1fx7f]{1,4})'
				| (?P<op>\.not\.|\.or\.|\.eor\.|\.and\.|<=|>=|<>|[-+*/!|=<>()])
				| (?P<id>[_A-Za-z][_A-Za-z0-9]*)
			)
		""", re.X | re.I)

		# 0.AND.(1/0) generates a Numeric Error in Operand
	BINARY = {

		'*': (3, _binary(lambda x,y: x*y)),
		'/': (3, _binary(lambda x,y: x//y)),
		'|': (3, _binary(orca_shift)),
		'!': (3, _binary(orca_shift)),

		'+': (4, _binary(lambda x,y: x+y)),
		'-': (4, _binary(lambda x,y: x-y)),
		'.or.': (4, _binary(lambda x,y: int(bool(x) or bool(y)))),
		'.eor.': (4, _binary(lambda x,y: int(bool(x) ^ bool(y)))),
		'.and.': (4, _binary(lambda x,y: int(bool(x) and bool(y)))),

		'=': (5, _binary(lambda x,y: int(x==y))),
		'<': (5, _binary(lambda x,y: int(x<y))),
		'>': (5, _binary(lambda x,y: int(x>y))),
		'<=': (5, _binary(lambda x,y: int(x<=y))),
		'>=': (5, _binary(lambda x,y: int(x>=y))),
		'<>': (5, _binary(lambda x,y: int(x!=y))),
	}

	UNARY = {
		'+': (2, _unary(lambda x: +x)),
		'-': (2, _unary(lambda x: -x)),
		'~': (2, _unary(lambda x: ~x)),
		'.not.': (2, _unary(lambda x: int(not x))),
	}

	def __init__(self):
		super(OrcaParser, self).__init__()


class MPWParser(Parser):

	name = 'MPW Asm'
	default_type = int32_t

	RE = re.compile(r"""
			(?:[ \t]*)
			(?:
				  \$(?P<hex>[A-Fa-f0-9]+)
				| %(?P<bin>[01]+)
				| (?P<dec>[0-9]+)
				| '(?P<cc>[^'\x00-\x1fx7f]{1,4})'
				| (?P<op>not|div|mod|and|or|xor|eor|<=|>=|<>|<<|>>|\*\*|\+\+|--|//|[-+*/=<>()¬≈÷≠≤≥Ω])
				| (?P<id>[_A-Za-z][_A-Za-z0-9]*)
			)
		""", re.X | re.I)

		# division by 0 -> 0
		# "logical" ops aren't exactly.
	BINARY = {

		'*': (4, _binary(lambda x,y: x*y)),

		'+': (6, _binary(lambda x,y: x+y)),
		'-': (6, _binary(lambda x,y: x-y)),

		'/': (5, _binary(lambda x,y: y if not y else x//y )),
		'÷': (5, _binary(lambda x,y: y if not y else x//y )),
		'div': (5, _binary(lambda x,y: y if not y else x//y )),
		'//': (5, _binary(lambda x,y: y if not y else x%y )),
		'mod': (5, _binary(lambda x,y: y if not y else x%y )),

		'<<': (7, _binary(lambda x,y: x<<y)),
		'>>': (7, _binary(lambda x,y: x>>y)),
		'=': (7, _binary(lambda x,y: int(x==y))),
		'<': (7, _binary(lambda x,y: int(x<y))),
		'>': (7, _binary(lambda x,y: int(x>y))),
		'<=': (7, _binary(lambda x,y: int(x<=y))),
		'≤': (7, _binary(lambda x,y: int(x<=y))),
		'>=': (7, _binary(lambda x,y: int(x>=y))),
		'≥': (7, _binary(lambda x,y: int(x>=y))),
		'<>': (7, _binary(lambda x,y: int(x!=y))),
		'≠': (5, _binary(lambda x,y: int(x!=y))),


		'or': (9, _binary(lambda x,y: x | y)),
		'++': (9, _binary(lambda x,y: x | y)),
		'Ω': (9, _binary(lambda x,y: x | y)),
		'ω': (9, _binary(lambda x,y: x | y)), # lowercase....

		'eor': (9, _binary(lambda x,y: x ^ y)),
		'xor': (9, _binary(lambda x,y: x ^ y)),
		'--': (9, _binary(lambda x,y: x ^ y)),

		'and': (8, _binary(lambda x,y: x & y)),
		'**': (8, _binary(lambda x,y: x & y)),


	}

	UNARY = {
		'+': (3, _unary(lambda x: +x)),
		'-': (3, _unary(lambda x: -x)),
		'≈': (2, _unary(lambda x: ~x)),
		'not': (2, _unary(lambda x: x ^ 1)),
		'¬': (2, _unary(lambda x: x ^ 1)),
	}

	def __init__(self):
		super(MPWParser, self).__init__()


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

	print("{}".format(num))
	#for x in num.alternates() :
	#	print("{:24}".format(str(x)), end="")

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


class Evaluator(object):
	def __init__(self):
		self.p = CParser()
		self.env = {}
		self.env["_"] = Number(0, self.p.default_type)

	def repl(self):
		while True:
			s = ""
			try:
				s = input("] ")
				s.strip()
				if not s: continue

				if s[0] == ".": self.dot(s)
				else: self.ep(s)

			except EOFError as ex:
				print()
				return 0
			except KeyboardInterrupt as ex:
				print()
				return 0

			except KeyError as ex:
				print("Unbound variable: {}".format(ex.args[0]))

			except Exception as e:
				print(e)
				# raise

	def ep(self, s):
		x = self.p.evaluate(s, self.env)
		self.env["_"]=x
		display(x)

	LANG = {
		'cc': CParser,
		'c': CParser,
		'pascal': PascalParser,
		'merlin': MerlinParser,
		'orca': OrcaParser,
		'mpw': MPWParser,
	}
	def dot(self, s):

		if s == ".quit":
			raise EOFError()

		if s == ".clear":
			self.env = {}
			self.env["_"] = Number(0, self.p.default_type)
			return

		if s == ".lang":
			print("Language:", self.p.name)
			return
		if s == ".int":
			print("Integer:", self.p.default_type.name)

		m = re.match(r"\.lang\s+([A-Za-z]+)", s)
		if m:
			lang = m[1].lower()
			if lang in self.LANG:
				self.p = self.LANG[lang]()
				print("Language:", self.p.name)
			else:
				print("Bad language:", lang)
			return

		m = re.match(r"\.int\s+(\d+)", s)
		if m:
			n = int(m[0])
			if n in (16, 32, 64):
				pass
			else:
				print("Bad integer size:", n)
			return




if __name__ == '__main__':

	repl = Evaluator()
	repl.repl()
	exit(0)
