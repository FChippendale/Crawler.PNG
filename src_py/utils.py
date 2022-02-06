import numpy as np
from PIL import Image


def create_dummy_tile(direction, instruction):
    value = np.uint8(direction*32 + instruction)
    return value


def get_direction_offset(direction):
    directions = np.array([[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]], dtype=np.int32)
    return directions[direction]


def read_tile(x, y, world):
    data = world[x, y].squeeze()
    direction = np.int32(data // 32)
    instruction = np.int32(data % 32)
    return direction, instruction


def read_tile_data(x, y, world):
    direction, _ = read_tile(x, y, world)
    direction_offsets = get_direction_offset(((np.atleast_1d(direction)[:, None] + np.arange(0, 8, 2))%8)[None, :])
    centers = np.stack([np.atleast_1d(x), np.atleast_1d(y)], axis=1)
    data_positions = centers[:, None, :] + direction_offsets
    value_bytes = world[data_positions[..., 0], data_positions[..., 1]]
    return value_bytes


def write_tile_data(x, y, world, value):
    direction, _ = read_tile(x, y, world)
    direction_offsets = get_direction_offset(((np.atleast_1d(direction)[:, None] + np.arange(0, 8, 2))%8)[None, :])
    centers = np.stack([np.atleast_1d(x), np.atleast_1d(y)], axis=1)
    data_positions = centers[:, None, :] + direction_offsets
    byte_data = np.frombuffer(value.tobytes(), dtype=np.uint8).reshape(len(np.atleast_1d(value)), 1, 4)
    world[data_positions[..., 0], data_positions[..., 1]] = byte_data


def open_image(path):
    im = Image.open(path)
    world = np.flip(np.rot90(np.array(im), 3), axis=1)
    return world


def save_image(world, path):
    im = Image.fromarray(np.rot90(np.flip(world, axis=1)))
    im.save(path)
