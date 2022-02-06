import pytest
from src_py.functions import *
from src_py.utils import create_dummy_tile
import numpy as np


def test_evaluate():
    world = np.zeros((3, 3), dtype=np.uint8)
    
    world[1][1] = create_dummy_tile(0, 11)
    world[1][0] = create_dummy_tile(0, 1)
    assert evaluate(1, 1, world) == 1

    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(5, 14)
    world[2][1] = create_dummy_tile(7, 6)
    world[1][2] = create_dummy_tile(2, 0)
    world[0][1] = create_dummy_tile(2, 6)

    assert evaluate(1, 1, world) == np.float32(12345.67)


def test_write():
    assert True


def test_read_arrow():
    world = np.zeros((5, 5), dtype=np.uint8)
    
    world[2][2] = create_dummy_tile(0, 11)
    world[2][1] = create_dummy_tile(0, 1)

    world[3][4] = create_dummy_tile(0, 0)
    world[1][3] = create_dummy_tile(1, 0)
    world[0][3] = create_dummy_tile(2, 0)
    world[1][1] = create_dummy_tile(3, 0)
    world[1][0] = create_dummy_tile(4, 0)
    world[3][1] = create_dummy_tile(5, 0)
    world[4][1] = create_dummy_tile(6, 0)
    world[3][3] = create_dummy_tile(7, 0)
    
    assert evaluate(2, 2, world) == 1
    assert evaluate(3, 4, world) == 1
    assert evaluate(1, 3, world) == 1
    assert evaluate(0, 3, world) == 1
    assert evaluate(1, 1, world) == 1
    assert evaluate(1, 0, world) == 1
    assert evaluate(3, 1, world) == 1
    assert evaluate(4, 1, world) == 1
    assert evaluate(3, 3, world) == 1


def test_write_arrow():
    assert True


def test_addition():
    world = np.zeros((5, 5), dtype=np.uint8)

    world[2][2] = create_dummy_tile(1, 1)
    world[1][1] = create_dummy_tile(0, 11)
    world[3][3] = create_dummy_tile(0, 11)

    world[1][0] = create_dummy_tile(0, 1)
    world[3][2] = create_dummy_tile(0, 1)

    assert evaluate(2, 2, world) == 2
    assert isinstance(evaluate(2, 2, world), np.int32)

    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(0, 0)
    world[2][1] = create_dummy_tile(0, 0)
    world[1][2] = create_dummy_tile(4, 0)
    world[0][1] = create_dummy_tile(1, 31)

    assert evaluate(2, 2, world) == np.float32(2.0)
    assert isinstance(evaluate(2, 2, world), np.float32)


def test_subtraction():
    world = np.zeros((5, 5), dtype=np.uint8)

    world[2][2] = create_dummy_tile(1, 2)
    world[1][1] = create_dummy_tile(0, 11)
    world[3][3] = create_dummy_tile(0, 11)

    world[1][0] = create_dummy_tile(0, 4)
    world[3][2] = create_dummy_tile(0, 2)

    assert evaluate(2, 2, world) == 2
    assert isinstance(evaluate(2, 2, world), np.int32)

    # float 1.0
    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(0, 0)
    world[2][1] = create_dummy_tile(0, 0)
    world[1][2] = create_dummy_tile(4, 0)
    world[0][1] = create_dummy_tile(1, 31)

    assert evaluate(2, 2, world) == np.float32(-1.0)
    assert isinstance(evaluate(2, 2, world), np.float32)


def test_multiplication():
    world = np.zeros((5, 5), dtype=np.uint8)

    world[2][2] = create_dummy_tile(1, 3)
    world[1][1] = create_dummy_tile(0, 11)
    world[3][3] = create_dummy_tile(0, 11)

    world[1][0] = create_dummy_tile(0, 4)
    world[3][2] = create_dummy_tile(0, 2)

    assert evaluate(2, 2, world) == 8
    assert isinstance(evaluate(2, 2, world), np.int32)

    # float 2.0
    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(0, 0)
    world[2][1] = create_dummy_tile(0, 0)
    world[1][2] = create_dummy_tile(0, 0)
    world[0][1] = create_dummy_tile(2, 0)

    assert evaluate(2, 2, world) == np.float32(4.0)
    assert isinstance(evaluate(2, 2, world), np.float32)


def test_division():
    world = np.zeros((5, 5), dtype=np.uint8)

    world[2][2] = create_dummy_tile(1, 4)
    world[1][1] = create_dummy_tile(0, 11)
    world[3][3] = create_dummy_tile(0, 11)

    world[1][0] = create_dummy_tile(0, 4)
    world[3][2] = create_dummy_tile(0, 2)

    assert evaluate(2, 2, world) == np.float32(2)
    assert isinstance(evaluate(2, 2, world), np.float32)

    # float 2.0
    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(0, 0)
    world[2][1] = create_dummy_tile(0, 0)
    world[1][2] = create_dummy_tile(0, 0)
    world[0][1] = create_dummy_tile(2, 0)

    assert evaluate(2, 2, world) == np.float32(1.0)
    assert isinstance(evaluate(2, 2, world), np.float32)


def test_floor_division():
    world = np.zeros((5, 5), dtype=np.uint8)

    world[2][2] = create_dummy_tile(1, 5)
    world[1][1] = create_dummy_tile(0, 11)
    world[3][3] = create_dummy_tile(0, 11)

    world[1][0] = create_dummy_tile(0, 4)
    world[3][2] = create_dummy_tile(0, 3)
    
    assert evaluate(2, 2, world) == np.int32(1)
    assert isinstance(evaluate(2, 2, world), np.int32)

    # float 8.0
    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(0, 0)
    world[2][1] = create_dummy_tile(0, 0)
    world[1][2] = create_dummy_tile(0, 0)
    world[0][1] = create_dummy_tile(2, 1)

    assert evaluate(2, 2, world) == np.int32(2.0)
    assert isinstance(evaluate(2, 2, world), np.int32)


def test_modulo():
    world = np.zeros((5, 5), dtype=np.uint8)

    world[2][2] = create_dummy_tile(1, 6)
    world[1][1] = create_dummy_tile(0, 11)
    world[3][3] = create_dummy_tile(0, 11)

    world[1][0] = create_dummy_tile(0, 4)
    world[3][2] = create_dummy_tile(0, 3)
    
    assert evaluate(2, 2, world) == np.int32(1)
    assert isinstance(evaluate(2, 2, world), np.int32)

    # float 4.0
    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(0, 0)
    world[2][1] = create_dummy_tile(0, 0)
    world[1][2] = create_dummy_tile(4, 0)
    world[0][1] = create_dummy_tile(2, 0)

    assert evaluate(2, 2, world) == np.float32(1.0)
    assert isinstance(evaluate(2, 2, world), np.float32)


def test_not_tile():
    world = np.zeros((5, 5), dtype=np.uint8)

    # not(int(4) == int(4)) ?
    world[2][2] = create_dummy_tile(1, 8)
    world[3][1] = create_dummy_tile(5, 7)

    world[1][1] = create_dummy_tile(0, 11)
    world[3][3] = create_dummy_tile(0, 11)

    world[1][0] = create_dummy_tile(0, 4)
    world[3][2] = create_dummy_tile(0, 4)
    
    assert evaluate(3, 1, world) == False

    # not(not(float(4.0) == int(4))) ?
    world[1][1] = create_dummy_tile(0, 12)
    world[4][0] = create_dummy_tile(5, 7)

    world[1][0] = create_dummy_tile(0, 0)
    world[2][1] = create_dummy_tile(0, 0)
    world[1][2] = create_dummy_tile(4, 0)
    world[0][1] = create_dummy_tile(2, 0)

    assert evaluate(4, 0, world) == True


def test_equal_tile():
    world = np.zeros((5, 5), dtype=np.uint8)

    # int(4) == int(4) ?
    world[2][2] = create_dummy_tile(1, 8)
    world[1][1] = create_dummy_tile(0, 11)
    world[3][3] = create_dummy_tile(0, 11)

    world[1][0] = create_dummy_tile(0, 4)
    world[3][2] = create_dummy_tile(0, 4)
    
    assert evaluate(2, 2, world) == True

    # float(4.0) == int(4) ?
    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(0, 0)
    world[2][1] = create_dummy_tile(0, 0)
    world[1][2] = create_dummy_tile(4, 0)
    world[0][1] = create_dummy_tile(2, 0)

    assert evaluate(2, 2, world) == True


def test_greater_tile():
    world = np.zeros((5, 5), dtype=np.uint8)

    world[2][2] = create_dummy_tile(1, 9)
    world[1][1] = create_dummy_tile(0, 11)
    world[3][3] = create_dummy_tile(0, 11)

    world[1][0] = create_dummy_tile(0, 4)
    world[3][2] = create_dummy_tile(0, 3)
    
    assert evaluate(2, 2, world) == True

    # float 4.0
    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(0, 0)
    world[2][1] = create_dummy_tile(0, 0)
    world[1][2] = create_dummy_tile(4, 0)
    world[0][1] = create_dummy_tile(2, 0)

    assert evaluate(2, 2, world) == True


def test_lesser_tile():
    world = np.zeros((5, 5), dtype=np.uint8)

    world[2][2] = create_dummy_tile(1, 10)
    world[1][1] = create_dummy_tile(0, 11)
    world[3][3] = create_dummy_tile(0, 11)

    world[1][0] = create_dummy_tile(0, 4)
    world[3][2] = create_dummy_tile(0, 5)
    
    assert evaluate(2, 2, world) == True

    # float 4.0
    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(0, 0)
    world[2][1] = create_dummy_tile(0, 0)
    world[1][2] = create_dummy_tile(4, 0)
    world[0][1] = create_dummy_tile(2, 0)

    assert evaluate(2, 2, world) == True



def test_read_array():
    assert True


def test_write_array():
    assert True


def test_read_array_index():
    assert True


def test_write_array_index():
    assert True


def test_read_ptr():
    assert True


def test_write_ptr():
    assert True


def test_print_tile():
    assert True


def test_input_tile():
    assert True


def test_write_tile():
    assert True


def test_null_tile():
    assert True