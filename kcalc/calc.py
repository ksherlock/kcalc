#
# python 3

import readline
import re
from functools import reduce
from collections import OrderedDict
import traceback
import math

# WS = re.compile("[ \t]*")

from .number import Number, unary_op, binary_op, logical_or, logical_and, logical_xor
from .type import *

class NegativeDivisionError(ArithmeticError):
	def __init__(self, /, *args, **kwargs):
		super(NegativeDivisionError, self).__init__(*args, **kwargs)
		


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

def iso_pascal_modulo(a,b):

	# i mod j == (i - (k * j)) for integral k such that 0 <= i mod j < j
	# therefore, answer is always positive.
	# 10 mod 3 = 1; (k = 3) -10 mod 3 = 2 ( k = -4)
	# this corresponds to python %, aside from negative denominotor
	# (which is an error)

	if b < 0: raise NegativeDivisionError('integer modulo by negative')
	return a % b


def iso_pascal_division(a,b):

	# abs(i) - abs(j) < abs( (i div j) * j) <= abs(i)
	# positive if i & j have the same sign
	# otherwise, negative.

	# therefore must round toward 0.

	if (a < 0) == (b < 0): return a // b
	return -(abs(a) // abs(b))



def orca_shift(a,b):
	# -b is a shift right
	# signed shift.
	a = int32_t.cast(a)
	b = int32_t.cast(b)
	if b < 0: return a >> abs(b)
	return a << b 


class Parser(object):
	"""docstring for Parser"""
	default_int = None
	default_bool = None

	default_mutable = False

	def __init__(self):
		pass

	def help(self):
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
		if x: return Number(int(x, 10), self.default_int)
		x = data.get("hex", None);
		if x: return Number(int(x, 16), self.default_int)
		x = data.get("bin", None);
		if x: return Number(int(x, 2), self.default_int)
		x = data.get("oct", None);
		if x: return Number(int(x, 8), self.default_int)

		x = data.get("id", None);
		if x: return env[m["id"]]

		x = data.get("cc", None);
		if x:
			xx = [ord(y) for y in x]
			xx.reverse() # make little endian
			return Number(reduce(lambda x,y: (x << 8) + y, xx), self.default_int)

		raise Exception("Missing type...")



	def _token(self):
		if not self._tokens: return None
		return self._tokens.pop()

	def _peek_token(self):
		if not self._tokens: return None
		return self._tokens[-1]


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




#
# C TODO
# sizeof (1/0) is not an error. -- exceptions need to retain the type.
# << and >> use the lhs type
# promote numbers to the default integer in an expression, eg (int16_t)1 is 16-bit, +(int16_t)1 is 32-bit
class CParser(Parser):

	name = 'C'
	default_int = int32_t
	default_size_t = uint64_t
	default_mutable = True

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
				| \(\s*
					(?P<cast>
						u?int64_t|
						u?int32_t|
						u?int16_t|
						u?int8_t|
						size_t|
						(
							(signed|unsigned|char|short|int|long)
							(\s+(signed|unsigned|char|short|int|long))*
						)
					)
					\s*\)
				| (?P<op><<|>>|<=|>=|==|!=|&&|\|\||[-+=<>~*!~/%^&|()<>]|sizeof)
				| (?P<id>[_A-Za-z][_A-Za-z0-9]*)
				| '(?P<cc>
					(
						[^'\\\x00-\x1f\x7f] |
						\\ ['"?\\abfnrtv] |
						\\ ([0-7]{1,3}) |
						\\ [xX] ([0-9a-fA-F]{1,}) # no length limit. 
					)
					{1,4}
				)'
			)
		""", re.X)

	CHAR_DECODE = {
		'\'': 0x27,
		'\"': 0x22,
		'\\': 0x5c,
		'?': 0x3f,
		'a': 0x07,
		'b': 0x08,
		'f': 0x0c,
		'n': 0x0a,
		'r': 0x0d,
		't': 0x09,
		'v': 0x0b,
	}

	# todo - boolean ops should drop back to the default type.
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

		'&&': (11, logical_and),
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
		'sizeof': (2, lambda n: n if n._exception else Number(n._type.bits() >> 3)),
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


	def parse_type(self, q):

		q = q.strip()
		if not q: raise Exception("Missing type")

		if q == 'size_t': return self.default_size_t

		for x in Type:
			if x.name == q: return x

		a = re.split(r'\s+', q)

		unsigned = 0
		signed = 0
		c = 0
		s = 0
		i = 0
		l = 0
		ll = 0


		for x in a:
			if x == 'unsigned': unsigned += 1
			elif x == 'signed': signed += 1
			elif x == 'char': c += 1
			elif x == 'short': s += 1
			elif x == 'int': i += 1
			elif x == 'long': l += 1
			else: raise Exception("Bad type: {}".format(q))

		if l == 2: ll = 1; l = 0
		if signed + unsigned > 1: raise Exception("Bad type: {}".format(q))
		if i + c > 1: raise Exception("Bad type: {}".format(q))
		if c + s + l + ll > 1: raise Exception("Bad type: {}".format(q))

		if c + s + i + l + ll == 0: i = 1

		tp = None
		if c: tp = int8_t
		elif s: tp = int16_t
		elif l: tp = int32_t
		elif ll: tp = int64_t
		elif i: tp = self.default_int
		else: raise Exception("Bad type: {}".format(q))


		if unsigned: tp = tp.make_unsigned()
		return tp


	def parse_char_constant(self, s):
		# n.b. c++ 'a' is char, c 'a' is int.
		# c++ multichar string is int.

		# TODO -- throw if hex/octal constant out of range.
		# TODO -- error on bad escape seq?

		# TODO - \unnnn, \Unnnnnnnn escapes.
		# TODO - u8, u, U, L, prefixes.

		_oct = "01234567"
		_hex = "0123456789abcdefABCDEF"

		st = 0
		value = 0
		tmp = 0
		ucount = 0
		for c in s:

			if st == 2: # octal
				if c in _oct:
					tmp <<= 3
					tmp |= int(c, 8)
					continue
				value <<= 8
				value |= (tmp & 0xff)
				st = 0

			if st == 3: # hex
				if c in _hex:
					tmp <<= 4
					tmp |= int(c, 16)
					continue
				value <<= 8
				value |= (tmp & 0xff)
				st = 0

			if st == 1:
				if c in _oct:
					tmp = int(c, 8)
					st = 2
				elif c in "xX":
					tmp = 0
					st = 3
				elif c in "uU":
					st = 4
					tmp = 0
					ucount = 4 if c == "u" else 8
				else:
					value <<= 8
					value |= self.CHAR_DECODE[c]
					st = 0
				continue

			if st == 0:
				if c == '\\':
					st = 1
				else:
					value <<= 8
					value |= ord(c)
				continue

			if st == 4: # \u
				tmp <<= 4
				tmp |= int(c, 16);
				ucount -= 1
				if not ucount:
					value |= (tmp & 0xff)
					st = 0
				continue


		#
		if st in (2,3):
			value <<= 8
			value |= (tmp & 0xff)

		return value


	def _convert_token(self, m, env):

		if m["op"]: return m["op"]

		if m["cc"]:
			return Number(self.parse_char_constant(m["cc"]))

		if m["id"]: return env[m["id"]]

		tp = m["cast"]
		if tp: return self.parse_type(tp)

		tp = self.default_int
		suffix = m["suffix"]
		if suffix:
			suffix = suffix.lower()
			if suffix not in self.SUFFIX:
				raise Exception("Invalid suffix: {}".format(suffix))
			tp = self.SUFFIX[suffix]
			if suffix == 'u':
				tp = self.default_int.make_unsigned()

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



	def _term(self):
		tk = self._peek_token()
		if type(tk) == Type:
			tp = self._token()
			# type cast
			x = self._term()
			if x == None: raise Exception("Syntax error: expected terminal, found EOF")
			return x.cast(tp)

		# special case for sizeof(type)
		if tk == 'sizeof':
			_ = self._token()
			tk = self._peek_token()
			if type(tk) == Type:
				tp = self._token()
			else:
				x = self._term()
				tp = x._type # todo - exceptional support.
			return Number(tp.bits() >> 3, self.default_size_t)

		return super()._term()


	def __init__(self):
		super(CParser, self).__init__()


class PascalParser(Parser):

	name = 'Pascal'
	default_int = int32_t
	default_mutable = True

	RE = re.compile(r"""
			(?:[ \t]*)
			(?:
				  \$(?P<hex>[A-Fa-f0-9]+)
				| %(?P<bin>[01]+)
				| (?P<dec>[0-9]+)
				| '(?P<cc>[^'\x00-\x1f\x7f]{1,4})'
				| (?P<op>div|mod|and|or|not|xor|<<|>>|<=|>=|<>|:=|[-+*=<>&|!~()])
				| (?P<id>[_A-Za-z][_A-Za-z0-9]*)
			)
		""", re.X | re.I)

	RELOPS = {
		'=': _binary(lambda x,y: x==y),
		'<': _binary(lambda x,y: x<y),
		'>': _binary(lambda x,y: x>y),
		'<=': _binary(lambda x,y: x<=y),
		'>=': _binary(lambda x,y: x>=y),
		'<>': _binary(lambda x,y: x!=y),
	}

	ADDOPS = {
		'+': _binary(lambda x,y: x+y),
		'-': _binary(lambda x,y: x-y),
		'|': _binary(lambda x,y: x|y),
		'!': _binary(lambda x,y: x^y),
		'xor': _binary(lambda x,y: x^y),
		'or': logical_or,
	}

	MULOPS = {
		'*': _binary(lambda x,y: x*y),
		'&': _binary(lambda x,y: x&y),
		'<<': _binary(lambda x,y: x<<y),
		'>>': _binary(lambda x,y: x>>y),
		'div': _binary(iso_pascal_division),
		'mod': _binary(iso_pascal_modulo),
		'and': logical_and,
	}
	SIGNOPS = {
		'+': _unary(lambda x: +x),
		'-': _unary(lambda x: -x),	
	}
	UNARYOPS = {
		'~': _unary(lambda x: ~x),
		'not': _unary(lambda x: not x),
	}

	# pascal unary +- are too weird for shunting yard.
	def _expr(self):
		a = self._simple_expr()
		tk = self._peek_token()
		while tk in self.RELOPS:
			tk = self._token()
			b = self._simple_expr()
			a = self.RELOPS[tk](a, b)
			tk = self._peek_token()
		return a

	def _simple_expr(self):
		a = self._term_expr(True)
		tk = self._peek_token()
		while tk in self.ADDOPS:
			tk = self._token()
			b = self._term_expr()
			a = self.ADDOPS[tk](a, b)
			tk = self._peek_token()
		return a

	def _term_expr(self, unary=False):

		sign = None
		if unary:
			tk = self._peek_token()
			if tk in self.SIGNOPS: sign = self._token()

		a = self._factor_expr()
		tk = self._peek_token()
		while tk in self.MULOPS:
			tk = self._token()
			b = self._factor_expr()
			a = self.MULOPS[tk](a, b)
			tk = self._peek_token()

		if sign:
			a = self.SIGNOPS[sign](a)
		return a


	def _factor_expr(self):
		unaryops = ('~', 'not')

		tk = self._token()
		if type(tk) == Number: return tk

		if tk in self.UNARYOPS:
			a = self._factor_expr()
			return self.UNARYOPS[tk](a)

		if tk == '(':
			a = self._expr()
			tk = self._token()
			if tk == ')': return a
			if tk == None: tk = "EOF"
			raise Exception("Syntax error: expected ), found {}".format(tk))

		if tk == None: tk = "EOF"
		raise Exception("Syntax error: expected terminal, found {}".format(tk))


	def __init__(self):
		super(PascalParser, self).__init__()


class MerlinParser(Parser):

	name = 'Merlin'
	default_int = uint32_t

	# should also have "x" for high-ascii char constant.
	RE = re.compile(r"""
			(?:[ \t]*)
			(?:
				  \$(?P<hex>[A-Fa-f0-9]+)
				| %(?P<bin>[01_]+)
				| (?P<dec>[0-9]+)
				| '(?P<cc>[^'\x00-\x1f\x7f]{1,4})'
				| "(?P<cch>[^"\x00-\x1f\x7f]{1,4})"
				| (?P<op>[-+*/!.&<>=])
				| (?P<id>[_A-Za-z][_A-Za-z0-9]*)
			)
		""", re.X | re.I)

	BINARY = {
		'*': (1, _binary(lambda x,y: x*y)),
		'/': (1, _binary(lambda x,y: -1 if y == 0 else x//y )),
		'+': (1, _binary(lambda x,y: x+y)),
		'-': (1, _binary(lambda x,y: x-y)),
		'&': (1, _binary(lambda x,y: x&y)),
		'.': (1, _binary(lambda x,y: x|y)),
		'!': (1, _binary(lambda x,y: x^y)),
		'>': (1, _binary(lambda x,y: x>y)),
		'<': (1, _binary(lambda x,y: x<y)),
		'=': (1, _binary(lambda x,y: x==y)),
	}

	UNARY = {
		'+': (1, _unary(lambda x: +x)),
		'-': (1, _unary(lambda x: -x)),	
	}


	def _convert_token(self, m, env):

		x = m["bin"]
		if x:
			x = x.replace('_', '')
			if not x: raise Exception("Syntax error")
			return Number(int(x, 2), self.default_int)

		# x = m["hex"]
		# if x:
		# 	x = x.replace('_', '')
		# 	if not x: raise Exception("Syntax error")
		# 	return Number(int(x, 16), self.default_int)


		x = m["cch"]
		if x:
			xx = [ord(y) | 0x80 for y in x]
			xx.reverse() # make little endian
			return Number(reduce(lambda x,y: (x << 8) + y, xx), self.default_int)

		return super()._convert_token(m, env)


	def __init__(self):
		super(MerlinParser, self).__init__()

	def help(self):
		print(
			"Merlin Assembler",
			"----------------",
			" +            Unary no-op",
			" -            Unary negation",
			" *            Multiplication",
			" /            Division",
			" +            Addition",
			" -            Subtraction",
			" &            Bitwise AND",
			" .            Bitwise OR",
			" !            Bitwise exclusive-OR",
			" >            Greater than",
			" <            Less than",
			" =            Equal to",

			sep = "\n"
		)


class OrcaParser(Parser):

	name = 'ORCA/M'
	default_int = uint32_t

	RE = re.compile(r"""
			(?:[ \t]*)
			(?:
				  \$(?P<hex>[A-Fa-f0-9]+)
				| %(?P<bin>[01]+)
				| (?P<dec>[0-9]+)
				| @(?P<oct>[0-7]+)
				| '(?P<cc>[^'\x00-\x1f\x7f]{1,4})'
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
		'.or.': (4, _binary(lambda x,y: bool(x) or bool(y))),
		'.eor.': (4, _binary(lambda x,y: bool(x) ^ bool(y))),
		'.and.': (4, _binary(lambda x,y: bool(x) and bool(y))),

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
		'.not.': (2, _unary(lambda x: not x)),
	}

	def __init__(self):
		super(OrcaParser, self).__init__()


class MPWParser(Parser):

	name = 'MPW Asm'
	default_int = int32_t

	RE = re.compile(r"""
			(?:[ \t]*)
			(?:
				  \$(?P<hex>[A-Fa-f0-9]+)
				| %(?P<bin>[01]+)
				| (?P<dec>[0-9]+)
				| '(?P<cc>[^'\x00-\x1f\x7f]{1,4})'
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
		'=': (7, _binary(lambda x,y: x==y)),
		'<': (7, _binary(lambda x,y: x<y)),
		'>': (7, _binary(lambda x,y: x>y)),
		'<=': (7, _binary(lambda x,y: x<=y)),
		'≤': (7, _binary(lambda x,y: x<=y)),
		'>=': (7, _binary(lambda x,y: x>=y)),
		'≥': (7, _binary(lambda x,y: x>=y)),
		'<>': (7, _binary(lambda x,y: x!=y)),
		'≠': (5, _binary(lambda x,y: x!=y)),


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

	def help(self):
		print(
			"MPW Assembler",
			"-------------",
			" ()           Grouping",
			" ≈            One's complement",
			" ¬  NOT       Logical NOT",
			" -            Unary negation",
			" *            Multiplication",
			" /  DIV ÷     Division",
			" // MOD ÷     Modulus division",
			" +            Addition",
			" -            Subtraction",
			" >>           Shift right",
			" <<           Shift left",
			" =            Equal to",
			" <> ≠         Not equal to",
			" <            Less than",
			" >            Greater than",
			" <= ≤         Less than or equal to",
			" >= ≥         Greater than or equal to",
			" ** AND       AND",
			" ++ OR  Ω     OR", # MPW Asm IIgs says Ω ; MPW Asm says |
			" -- XOR EOR   Exclusive-OR\n",
			sep = "\n"
		)

	def __init__(self):
		super(MPWParser, self).__init__()



class NLParser(Parser):
	name = "Nifty List"
	default_int = int32_t
	RE = re.compile(r"""
			(?:[ \t]*)
			(?:
				  (?P<hex>[A-Fa-f0-9]+)
				| \#(?P<dec>[0-9]+)
				| "(?P<cc>[^"\x00-\x1f\x7f]{1,4})"
				| (?P<op>[-+*])
				| \#(?P<id>[_A-Za-z][_A-Za-z0-9]*)
			)
		""", re.X | re.I)

	UNARY = {
		'-': (1, _unary(lambda x: -x)),
	}
	BINARY = {
		'+': (1, _binary(lambda x,y: x+y)),
		'-': (1, _binary(lambda x,y: x-y)),
		'*': (1, _binary(lambda x,y: x-y)),
	}


	def __init__(self):
		super(NLParser, self).__init__()

class Evaluator(object):
	def __init__(self):
		self.p = CParser()
		self.little_endian = True
		self.env = {}
		self.env["_"] = Number(0, self.p.default_int)

	def repl(self, debug=False):
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
				if debug:
					traceback.print_exc() 
				else: print(e)

	def to_b(self, v, bits=32):
		rv = ""
		for i in range(0,bits):
			rv += "01"[v & 0x01]
			v >>= 1
		return rv[::-1]

	def to_cc(self, v):
		if not v: return ""
		b = []
		while v:
			b.append(v & 0xff)
			v >>= 8
		if not all([x>=0x20 and x<0x7f for x in b]): return ""
		if not self.little_endian: b.reverse()
		return "'" + "".join( [chr(x) for x in b] ) + "'"

	def to_hex(self, v, bits=32):
		rv = []
		for i in range(0,bits//8):
			rv.append(v & 0xff)
			v >>= 8
		if not self.little_endian: rv.reverse()
		return " ".join( ["{:02x}".format(x) for x in rv])


	def display(self, num):
		# 0xffff 0b1111 'cccc'
		value = num.value()
		uvalue = num.unsigned_value()
		tp = num.type()
		print("0x{:08x}  0b{:032b} {}".format(uvalue, uvalue, self.to_cc(uvalue)))

		tl = (uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t)

		dd = OrderedDict()
		dd[value] = tp
		for tt in tl:
			if tt.bits() > tp.bits(): continue
			if tt == tp: continue
			vv = num.unwrap_as(tt)
			dd.setdefault(vv,tt)

		for n, tt in dd.items():
			print("({:>8}): {}".format(tt.name, n))


		# int32_t, int16_t, int8_t: -1
		# uint32_t: ..


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

		# print()



	def ep(self, s):
		x = self.p.evaluate(s, self.env)
		self.env["_"]=x
		self.display(x)

	LANG = {
		'cc': CParser,
		'c': CParser,
		'pascal': PascalParser,
		'merlin': MerlinParser,
		'orca': OrcaParser,
		'mpw': MPWParser,
		'nl': NLParser,
	}
	WORD = {
		16: int16_t,
		32: int32_t,
		64: int64_t,
	}

	def dot_help(self, argv):

		self.p.help();
		print()
		print(
			".help                               - you are here",
			".lang [c|pascal|merlin|orca|mpw|nl] - set language",
			".word [16|32|64]                    - set word size",
			".ascii                              - print ASCII chart",
			".clear                              - clear variables",
			"",
			sep = "\n"
		)


	def dot_ascii(self, argv):
		print(
			"00 nul   01 soh   02 stx   03 etx   04 eot   05 enq   06 ack   07 bel",
			"08 bs    09 ht    0a lf    0b vt    0c ff    0d cr    0e so    0f si",
			"10 dle   11 dc1   12 dc2   13 dc3   14 dc4   15 nak   16 syn   17 etb",
			"18 can   19 em    1a sub   1b esc   1c fs    1d gs    1e rs    1f us",
			"20 sp    21  !    22  \"    23  #    24  $    25  %    26  &    27  '",
			"28  (    29  )    2a  *    2b  +    2c  ,    2d  -    2e  .    2f  /",
			"30  0    31  1    32  2    33  3    34  4    35  5    36  6    37  7",
			"38  8    39  9    3a  :    3b  ;    3c  <    3d  =    3e  >    3f  ?",
			"40  @    41  A    42  B    43  C    44  D    45  E    46  F    47  G",
			"48  H    49  I    4a  J    4b  K    4c  L    4d  M    4e  N    4f  O",
			"50  P    51  Q    52  R    53  S    54  T    55  U    56  V    57  W",
			"58  X    59  Y    5a  Z    5b  [    5c  \\    5d  ]    5e  ^    5f  _",
			"60  `    61  a    62  b    63  c    64  d    65  e    66  f    67  g",
			"68  h    69  i    6a  j    6b  k    6c  l    6d  m    6e  n    6f  o",
			"70  p    71  q    72  r    73  s    74  t    75  u    76  v    77  w",
			"78  x    79  y    7a  z    7b  {    7c  |    7d  }    7e  ~    7f del",
			sep="\n"
		)

	def dot_clear(self, argv):
		self.env = {}
		self.env["_"] = Number(0, self.p.default_int)

	def dot_quit(self, argv):
		raise EOFError()


	def dot_lang(self, argv):
		if len(argv) > 1:
			print(".lang [language]")
			return
		if len(argv) == 1:
			lang = argv[0].lower()

			if lang in self.LANG:
				self.p = self.LANG[lang]()

				t = self.p.default_int
				Number.set_default_int(t)
				Number.set_default_bool(t)
			else:
				print("Bad language:", lang)
				return

		print("Language:", self.p.name)

	def dot_word(self, argv):
		if len(argv) > 1:
			print(".word [size]")
			return
		if len(argv) == 1:

			if not self.p.default_mutable:
				print("Word size not mutable")
				return

			n = argv[0]
			if not n.isdigit():
				print(".word [size]")
				return
			n = int(n, 10)

			if n in self.WORD:
				t = self.WORD[n]
				Number.set_default_int(t)
				Number.set_default_bool(t)
				self.p.default_int = t
			else:
				print("Invalid word size:", n)
				return

		print("Word:", self.p.default_int.name)


	def dot_factor(self, argv):
		# factor a number...
		if len(argv) != 1:
			print(".factor number")
			return

		n = int(argv[0], 10) # todo -- be smarter....

		n = abs(n)
		if n < 3:
			print("prime.")
			return

		factors = []
		while (n & 0x01) == 0:
			n >>= 1
			factors.append(2)

		limit = math.floor(math.sqrt(n))+1
		for denom in range(3,limit, 2):
			while n % denom == 0: #== (n // denom) * denom:
				factors.append(denom)
				n //= denom
			if denom >= n: break

		if len(factors):
			if n>1: factors.append(n)
			print(" * ".join([str(x) for x in factors]))
		else:
			print("prime.")


	def dot_magic_divisor(self, argv):
		# hacker's delight magic numbers for division via multiplication.
		# see also
		# https://chromium.googlesource.com/v8/v8/+/1210d0c/src/base/division-by-constant.cc
		if len(argv) != 1:
			print(".magic denominator")
			return

		d = int(argv[0], 10) # todo -- be smarter....

		bits = Number.get_default_int().bits()
		mu = self.dot_magic_unsigned(d, bits)
		ms = self.dot_magic_signed(d, bits)

		magic, shift, addend = mu
		if addend:
			print("unsigned: (((n * {}) >> {}) + n) >> {}".format(
				magic, bits, shift 
			))
		else:
			print("unsigned: (n * {}) >> {}) >> {}".format(
				magic, bits, shift
			))
		# add 1 if n is negative???

		if mu != ms:
			magic, shift, _ = ms

			print("signed:   (n * {}) >> {}) >> {}".format(
				magic, bits, shift
			))



	def dot_magic_unsigned(self, d, bits):

		two31 = 1 << (bits-1)
		two32 = 1 << bits
		two31m1 = two31-1


		addind = False
		nc = two32 - 1 - (two32 - d) % d

		p = bits - 1
		q1 = two31 // nc
		r1 = two31 - (q1 * nc)
		q2 = two31m1 // d
		r2 = two31m1 - (q2*d)

		while True:

			p += 1
			if r1 > (nc - r1):
				q1 *= 2
				r1 *= 2
				q1 += 1
				r1 -= nc
			else:
				q1 *= 2
				r1 *= 2

			if (r2 + 1) >= (d - r2):
				if q2 >= two31m1: addind = True
				q2 *= 2
				r2 *= 2
				q2 += 1
				r2 += 1 - d
			else:
				if q2 >= two31: addind = True
				q2 *= 2
				r2 *= 2
				r2 += 1

			delta = d - 1 - r2

			if p >= (bits * 2): break
			if q1 < delta: continue
			if (q1 == delta) and (r1 == 0): continue
			break

		magic = q2 + 1
		if magic >= two32: magic -= two32
		shift = p - bits
		return magic, shift, addind



	def dot_magic_signed(self, d, bits):

		two31 = 1 << (bits-1)
		two32 = 1 << bits
		two31m1 = two31-1

		ad = abs(d)

		t = two31 + (d >> (bits - 1))
		anc = t - 1 - t%ad
		p = bits - 1
		q1 = two31 // anc
		r1 = two31 - (q1 * anc)
		q2 = two31 // ad
		r2 = two31 - (q2 * ad)


		while True:
			p += 1
			q1 *= 2
			r1 *= 2
			if r1 >= anc:
				q1 += 1
				r1 -= anc
			q2 *= 2
			r2 *= 2
			if r2 >= ad:
				q2 += 1
				r2 -= ad
			delta = ad - r2

			if q1 < delta: continue
			if (q1 == delta) and (r1 == 0): continue
			break


		magic = q2 + 1
		if d < 0: magic = two32 - magic
		shift = p - bits
		return magic, shift, False





	DOT_FUNCTIONS = {
		'.lang': dot_lang,
		'.help': dot_help,
		'.clear': dot_clear,
		'.word': dot_word,
		'.ascii': dot_ascii,
		'.magic': dot_magic_divisor,
		'.factor': dot_factor,
	}


	def build_argv(self, s):

		argv = []
		s = s.strip()
		scratch = None
		st = None

		for c in s:

			if st:
				if c == st:
					st = None
					continue
				scratch += c
				continue

			if c in "'\"":
				st = c
				if not scratch: scratch = ""
				continue

			if c.isspace():
				if scratch:
					argv.append(scratch)
					scratch = None
				continue

			if not scratch: scratch = ""
			scratch += c

		if st: raise Exception("Unterminated string")
		if scratch: argv.append(scratch)
		return argv


	def dot(self, s):

		[command, *argv] = self.build_argv(s)

		if command in self.DOT_FUNCTIONS:
			self.DOT_FUNCTIONS[command](self, argv)
		else:
			raise RuntimeError("unknown command: {}".format(command))

def main():
	import argparse

	go = argparse.ArgumentParser()
	go.add_argument('-g', '--debug', dest='debug', action='store_true')
	opts = go.parse_args()

	repl = Evaluator()
	repl.repl(opts.debug)
	exit(0)

