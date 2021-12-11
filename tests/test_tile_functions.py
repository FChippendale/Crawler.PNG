import pytest
from python_interpreter.tile_functions import *
from python_interpreter.utils import create_dummy_tile
import numpy as np


def test_int_tile():
    world = np.zeros((3, 3, 8), dtype=bool)
    
    world[1][1] = create_dummy_tile(0, 11)
    world[1][0] = create_dummy_tile(0, 1)
    assert int_tile(1, 1, world) == 1

    world[1][0] = create_dummy_tile(7, 31)
    world[1][2] = create_dummy_tile(7, 31)
    world[0][1] = create_dummy_tile(7, 31)
    world[2][1] = create_dummy_tile(7, 31)
    assert int_tile(1, 1, world) == -1


def test_float_tile():
    world = np.zeros((3, 3, 8), dtype=bool)
    
    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(5, 14)
    world[2][1] = create_dummy_tile(7, 6)
    world[1][2] = create_dummy_tile(2, 0)
    world[0][1] = create_dummy_tile(2, 6)

    assert float_tile(1, 1, world) == np.float32(12345.67)

    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(5, 14)
    world[2][1] = create_dummy_tile(7, 6)
    world[1][2] = create_dummy_tile(2, 0)
    world[0][1] = create_dummy_tile(6, 6)

    assert float_tile(1, 1, world) == np.float32(-12345.67)


def test_evaluate():
    world = np.zeros((3, 3, 8), dtype=bool)
    
    world[1][1] = create_dummy_tile(0, 11)
    world[1][0] = create_dummy_tile(0, 1)
    assert evaluate(1, 1, world) == 1

    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(5, 14)
    world[2][1] = create_dummy_tile(7, 6)
    world[1][2] = create_dummy_tile(2, 0)
    world[0][1] = create_dummy_tile(2, 6)

    assert evaluate(1, 1, world) == np.float32(12345.67)


def test_arrow_tile():
    world = np.zeros((5, 5, 8), dtype=bool)
    
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


def test_addition_tile():
    world = np.zeros((5, 5, 8), dtype=bool)

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


def test_subtraction_tile():
    world = np.zeros((5, 5, 8), dtype=bool)

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


def test_multiplication_tile():
    world = np.zeros((5, 5, 8), dtype=bool)

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


def test_division_tile():
    world = np.zeros((5, 5, 8), dtype=bool)

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


def test_floor_division_tile():
    world = np.zeros((5, 5, 8), dtype=bool)

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


def test_modulo_tile():
    world = np.zeros((5, 5, 8), dtype=bool)

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


def test_equal_tile():
    world = np.zeros((5, 5, 8), dtype=bool)

    world[2][2] = create_dummy_tile(1, 8)
    world[1][1] = create_dummy_tile(0, 11)
    world[3][3] = create_dummy_tile(0, 11)

    world[1][0] = create_dummy_tile(0, 4)
    world[3][2] = create_dummy_tile(0, 4)
    
    assert evaluate(2, 2, world) == True

    # float 4.0
    world[1][1] = create_dummy_tile(0, 12)

    world[1][0] = create_dummy_tile(0, 0)
    world[2][1] = create_dummy_tile(0, 0)
    world[1][2] = create_dummy_tile(4, 0)
    world[0][1] = create_dummy_tile(2, 0)

    assert evaluate(2, 2, world) == False


def test_greater_tile():
    world = np.zeros((5, 5, 8), dtype=bool)

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
    world = np.zeros((5, 5, 8), dtype=bool)

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


def test_not_tile():
    world = np.zeros((5, 5, 8), dtype=bool)

    world[2][2] = create_dummy_tile(1, 8)
    world[3][1] = create_dummy_tile(5, 7)

    world[1][1] = create_dummy_tile(0, 11)
    world[3][3] = create_dummy_tile(0, 11)

    world[1][0] = create_dummy_tile(0, 4)
    world[3][2] = create_dummy_tile(0, 4)
    
    assert evaluate(3, 1, world) == False

    # float 4.0
    world[1][1] = create_dummy_tile(0, 12)
    world[4][0] = create_dummy_tile(5, 7)

    world[1][0] = create_dummy_tile(0, 0)
    world[2][1] = create_dummy_tile(0, 0)
    world[1][2] = create_dummy_tile(4, 0)
    world[0][1] = create_dummy_tile(2, 0)

    assert evaluate(4, 0, world) == False

