
from type import *



default_type = int32_t


class Number(object):
	__slots__  = '_value', '_type', '_exception'


	def __init__(self, *args):
		if len(args) == 1:
			rhs = args[0]
			if type(rhs) == int:
				self._value = default_type.cast(rhs)
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
			v, t = args
			self._value = t.cast(v)
			self._type = t
			self._exception = None
		else:
			raise TypeError('Number() takes 1 or 2 arguments')

	def __str__(self):
		if (self._exception):
			return str(self._exception)
		return "({}){}".format(self._type.name, self._value)

	def value(self):
		if self._exception: raise self._exception
		return self._value

	def type(self):
		if self._exception: raise self._exception
		return self._type

	def is_signed(self):
		if self._exception: raise self._exception
		return (self._type & 0x01) == 1

	def signed_value(self):
		if self._exception: raise self._exception
		return self._type.make_signed().cast(self._value)

	def unsigned_value(self):
		if self._exception: raise self._exception
		return self._type.make_unsigned().cast(self._value)

	def unwrap_as(self, t):
		if self._exception: raise self._exception
		if self._type == t: return self._value
		return t.cast(self._value)

# todo -- exceptions should also have a type?
def unary_op(a, op):
	if a._exception: return a
	v = a._value
	t = a._type
	# if t < default_type:
	# 	t = default_type
	# 	v = coerce(v, t)
	try:
		return Number(+op(v), t)
	except Exception as e:
		return Number(e)

def binary_op(a, b, op):
	if a._exception: return a
	if b._exception: return b
	t = Type.common_type(a._type, b._type)
	# if t < default_type: t = default_type
	aa = t.cast(a._value) # need to coerce if swapping from int32 -> uint32
	bb = t.cast(b._value)

	try:
		return Number(+op(aa,bb), t)
	except Exception as e:
		return Number(e)



# special case for short-circuiting logical ops.
def logical_or(a, b):
	if a._exception: return a
	if a._value: return Number(1, a._type)
	if b._exception: return b
	return Number(int(b._value != 0), b._type)

def logical_and(a, b):
	if a._exception: return a
	if not a._value: return Number(0, a._type)
	if b._exception: return b
	return Number(int(b._value != 0), b._type)

def logical_xor(a, b):
	if a._exception: return a
	if b._exception: return b
	return Number(int(bool(a._value) ^ bool(b._value)), a._type)

