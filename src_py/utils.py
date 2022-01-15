import numpy as np


def create_dummy_tile(direction, instruction):
    value = np.uint8(direction*32 + instruction)
    return np.unpackbits(value)


def get_direction_offset(direction):
    directions = np.array([[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]], dtype=np.int32)
    return directions[direction]


def read_tile(x, y, world):
    bits = world[x, y]
    data = np.packbits(bits, axis=-1).squeeze()
    direction = data // 32
    instruction = data % 32
    return direction, instruction

