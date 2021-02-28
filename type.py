
from enum import Enum

# suffix_map = {
# 	'': default_type,
# 	'u': make_unsigned(default_type),
# 	'l': int32_t,
# 	'll': int64_t,
# 	'ul': uint32_t
# 	'lu': uint32_t,
# 	'ull': uint64_t,
# 	'llu': uint64_t,
# }

_range_table = (
	None, None,
	range(-0x80, 0x80),
	range(-0x00, 0x100),
	range(-0x8000, 0x8000),
	range(-0x0000, 0x10000),
	range(-0x80000000, 0x80000000),
	range(-0x00000000, 0x100000000),
	range(-0x8000000000000000, 0x8000000000000000),
	range(-0x0000000000000000, 0x10000000000000000),
)


class Type(Enum):
	int8_t   = 2
	uint8_t  = 3
	int16_t  = 4
	uint16_t = 5
	int32_t  = 6
	uint32_t = 7
	int64_t  = 8
	uint64_t = 9

	def range(self):
		return _range_table[self.value]

	def is_signed(self):
		return self.value & 0x01 == 0

	def make_signed(self):
		return Type(self.value & ~0x01)

	def make_unsigned(self):
		return Type(self.value | 0x01)

	@staticmethod
	def common_type(*args):
		return Type(max([x.value for x in args]))

	@staticmethod
	def type_for_value(value, /, base=None, signed=True, unsigned=True):
		start = 2
		step = 1
		# bits = (signed << 2) | (unsigned << 1)
		# if bits | (x & 0x01) in (0b010, 0b101): continue

		if not (signed | unsigned): return None
		if signed ^ unsigned: step = 2

		if base: start = base.value
		if not signed: start += 1

		for x in range(start,10,step):
			if value in _range_table[x]:
				return Type(x)

		return None

if __name__ == '__main__':
	int8_t = Type.int8_t
	uint8_t = Type.uint8_t
	int32_t = Type.int32_t
	uint32_t = Type.uint32_t
	print(int8_t)
	print(int32_t)
	print(int8_t.is_signed())
	print(uint8_t.is_signed())
	print(int8_t.make_unsigned())
	print(int32_t.make_unsigned())
	print(int8_t.range())
	print(int32_t.range())
	print(uint32_t.range())
	print(Type.common_type(int8_t, uint8_t))
	print(Type.common_type(uint8_t, int32_t))
	print()
	print(Type.type_for_value(32767))
	print(Type.type_for_value(32768, unsigned=False))
	print(Type.type_for_value(32768, signed=False))
	print(Type.type_for_value(-1, base=int32_t))