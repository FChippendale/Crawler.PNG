import pytest
from src_py.types import *
from src_py.utils import create_dummy_tile
import numpy as np


def test_read_int():
    world = np.zeros((3, 3), dtype=np.uint8)
    
    world[1][1] = create_dummy_tile(0, 11)
    world[1][0] = create_dummy_tile(0, 1)
    assert read_int(1, 1, world) == 1

    world[1][0] = create_dummy_tile(7, 31)
    world[1][2] = create_dummy_tile(7, 31)
    world[0][1] = create_dummy_tile(7, 31)
    world[2][1] = create_dummy_tile(7, 31)
    assert read_int(1, 1, world) == -1


def test_write_int():
    assert True


def test_read_float():
    world = np.zeros((3, 3), dtype=np.uint8)
    
    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(5, 14)
    world[2][1] = create_dummy_tile(7, 6)
    world[1][2] = create_dummy_tile(2, 0)
    world[0][1] = create_dummy_tile(2, 6)

    assert read_float(1, 1, world) == np.float32(12345.67)

    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(5, 14)
    world[2][1] = create_dummy_tile(7, 6)
    world[1][2] = create_dummy_tile(2, 0)
    world[0][1] = create_dummy_tile(6, 6)

    assert read_float(1, 1, world) == np.float32(-12345.67)


def test_write_float():
    assert True


def test_read_char():
    assert True


def test_write_char():
    assert True