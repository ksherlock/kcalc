
int8_t   = 2
uint8_t  = 3
int16_t  = 4
uint16_t = 5
int32_t  = 6
uint32_t = 7
int64_t  = 8
uint64_t = 9

_valid_ranges = [
	None,
	None,
	range(-0x80, 0x80),
	range(-0x00, 0x100),
	range(-0x8000, 0x8000),
	range(-0x0000, 0x10000),
	range(-0x80000000, 0x80000000),
	range(-0x00000000, 0x100000000),
	range(-0x8000000000000000, 0x8000000000000000),
	range(-0x0000000000000000, 0x10000000000000000),
]

_name_table = [
	None,
	None,
	'int8_t', 'uint8_t',
	'int16_t', 'uint16_t',
	'int32_t', 'uint32_t',
	'int64_t', 'uint64_t',
]

def coerce_uint8_t(x): return x & 0xff
def coerce_uint16_t(x): return x & 0xffff
def coerce_uint32_t(x): return x & 0xffffffff
def coerce_uint64_t(x): return x & 0xffffffffffffffff

def coerce_int8_t(x):
	x &= 0xff
	if x & 0x80: x -= 0x100
	return x

def coerce_int16_t(x):
	x &= 0xffff
	if x & 0x8000: x -= 0x10000
	return x

def coerce_int32_t(x):
	x &= 0xffffffff
	if x & 0x80000000: x -= 0x100000000
	return x

def coerce_int64_t(x):
	x &= 0xffffffffffffffff
	if x & 0x8000000000000000: x -= 0x10000000000000000
	return x


_coercion_table = [
	None,
	None,
	coerce_int8_t,
	coerce_uint8_t,
	coerce_int16_t,
	coerce_uint16_t,
	coerce_int32_t,
	coerce_uint32_t,
	coerce_int64_t,
	coerce_uint64_t,
]

def coerce(x, t):
	if x in _valid_ranges[t]: return x
	return _coercion_table[t](x)


def common_type(t1, t2):
	return max(t1, t2)

def make_unsigned(t1):
	return (t1 + 0) & ~0x01

default_type = int32_t


class Number(object):
	__slots__  = '_value', '_type', '_exception'


	def __init__(self, *args):
		if len(args) == 1:
			rhs = args[0]
			if type(rhs) == int:
				self._value = coerce(rhs, default_type)
				self._type = default_type
				self._exception = None
			elif type(rhs) == Number:
				self._value = rhs._value
				self._type = rhs._type
				self._exception = rhs._exception
			elif isinstance(rhs, Exception):
				self._type = None
				self._value = None
				self._exception = rhs
			else:
				raise TypeError("bad operand type for Number(): '{}'".format(type(rhs).__name__))

		elif len(args) == 2:
			self._value = coerce(args[0], args[1])
			self._type = args[1]
			self._exception = None
		else:
			raise TypeError('Number() takes 1 or 2 arguments')

	def __str__(self):
		if (self._exception):
			return str(self._exception)
		return "({}){}".format(_name_table[self._type], self._value)

	def value(self):
		if self._exception: raise self._exception
		return self._value

	def type(self):
		if self._exception: raise self._exception
		return self._type

	def is_signed(self):
		if self._exception: raise self._exception
		return (self._type & 0x01) == 1

	def unsigned_value(self):
		if self._exception: raise self._exception
		return _coercion_table[(self._type & ~0x1)+1](self._value)

	def unwrap_as(self, t):
		if self._exception: raise self._exception
		if self._type == t: return self._value
		return coerce(self._value, t)

	# build a list of unique alternate representations.
	def alternates(self):
		if self._exception: return [self]

		return [
			self,
			Number(self._value, self._type ^ 0x01)
		]

		# rv = []
		# unique = set()
		# start = make_unsigned(max(self._type, default_type))
		# for t in range(start, 0, -1):
		# 	n = Number(coerce(self._value, t),t)
		# 	if n._value in unique: continue
		# 	unique.add(n._value)
		# 	rv.append(n)
		# return rv


def unary_op(a, op):
	if a._exception: return a
	v = a._value
	t = a._type
	if t < default_type:
		t = default_type
		v = coerce(v, t)
	try:
		return Number(op(v), t)
	except Exception as e:
		return Number(e)

def binary_op(a, b, op):
	if a._exception: return a
	if b._exception: return b
	t = common_type(a._type, b._type)
	if t < default_type: t = default_type
	aa = coerce(a._value, t) # need to coerce if swapping from int32 -> uint32
	bb = coerce(b._value, t)

	try:
		return Number(op(aa,bb), t)
	except Exception as e:
		return Number(e)


# special case for short-circuiting logical ops.
def logical_or(a, b):
	if a._exception: return a
	if a._value: return Number(1, default_type)
	if b._exception: return b
	return Number(b._value != 0, default_type)

def logical_and(a, b):
	if a._exception: return a
	if not a._value: return Number(0, default_type)
	if b._exception: return b
	return Number(b._value != 0, default_type)

def logical_xor(a, b):
	if a._exception: return a
	if b._exception: return b
	return Number(int(bool(a._value) ^ bool(b._value)), default_type)

