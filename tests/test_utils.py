import pytest
from src_py.utils import *
import numpy as np


def test_get_direction_offset():
    assert np.array_equal(get_direction_offset(0), np.array([0, -1]))

    assert np.array_equal(
        get_direction_offset(np.array([0, 1, 2, 3])), 
        np.array([[0, -1], [1, -1], [1, 0], [1, 1]])
    )


def test_read_tile():
    world = np.zeros((1, 1, 8), dtype=bool)
    
    world[0][0] = np.unpackbits(np.uint8(1))
    direction, instruction = read_tile(0, 0, world)
    assert (direction, instruction) == (0, 1)

    world[0][0] = np.unpackbits(np.uint8(255))
    direction, instruction = read_tile(0, 0, world)
    assert (direction, instruction) == (7, 31)


def write_int(x, y, world, value):
    assert True


def write_float(x, y, world, value):
    assert True


def write_char(x, y, world, value):
    assert True