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
    world = np.zeros((1, 1), dtype=np.uint8)
    
    world[0][0] = np.uint8(1)
    direction, instruction = read_tile(0, 0, world)
    print(direction.shape, instruction.shape)
    assert (direction, instruction) == (0, 1)

    world[0][0] = np.uint8(255)
    direction, instruction = read_tile(0, 0, world)
    assert (direction, instruction) == (7, 31)


def test_read_tile_data():
    assert True


def test_write_tile_data():
    assert True