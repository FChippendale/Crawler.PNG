import numpy as np
from src_py.utils import get_direction_offset, read_tile
from src_py.types import *


def evaluate(x, y, world):
    evaluate_functions = {
        0:    read_arrow,
        1:    addition,
        2:    subtraction,
        3:    multiplication,
        4:    division,
        5:    floor_division,
        6:    modulo,
        7:    not_tile,
        8:    equal_tile,
        9:    greater_tile,
        10:   lesser_tile,
        11:   read_int,
        12:   read_float,
        13:   read_char,
        14:   read_ptr,
        15:   read_array,
        16:   print_tile,
        17:   input_tile,
        18:   write_tile,
        19:   read_array_index,

        31:   null_tile,
    }

    _, instruction = read_tile(x, y, world)
    if instruction in evaluate_functions:
        return evaluate_functions[instruction](x, y, world)


def write(x, y, world, value):
    write_functions = {
        0: write_arrow,
        11: write_int,
        12: write_float, 
        13: write_char,
        14: write_ptr,
        15: write_array,
        19: write_array_index,
    }
    _, instruction = read_tile(x, y, world)
    if instruction in write_functions:
        write_functions[instruction](x, y, world, value)
        return

    raise TypeError("Attempted to write to tile that could not be written to")


def read_arrow(x, y, world):
    direction, _ = read_tile(x, y, world)
    move_x, move_y = get_direction_offset(direction)
    return evaluate(x+move_x, y+move_y, world)


def write_arrow(x, y, world, value):
    direction, _ = read_tile(x, y, world)
    move_x, move_y = get_direction_offset(direction)
    return write(x+move_x, y+move_y, world, value)


def addition(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert A.dtype == np.int32 or A.dtype == np.float32, 'Addition is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert B.dtype == np.int32 or B.dtype == np.float32, 'Addition is only permitted using types: "Int", "Float"'
    
    if A.dtype != B.dtype:
        return (A + B).astype(np.float32)

    return A + B


def subtraction(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert A.dtype == np.int32 or A.dtype == np.float32, 'Subtraction is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert B.dtype == np.int32 or B.dtype == np.float32, 'Subtraction is only permitted using types: "Int", "Float"'
    
    if A.dtype != B.dtype:
        return (A - B).astype(np.float32)

    return A - B


def multiplication(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert A.dtype == np.int32 or A.dtype == np.float32, 'Multiplication is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert B.dtype == np.int32 or B.dtype == np.float32, 'Multiplication is only permitted using types: "Int", "Float"'
    
    if A.dtype != B.dtype:
        return (A * B).astype(np.float32)

    return A * B


def division(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert A.dtype == np.int32 or A.dtype == np.float32, 'Division is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert B.dtype == np.int32 or B.dtype == np.float32, 'Division is only permitted using types: "Int", "Float"'
    
    return (A / B).astype(np.float32)


def floor_division(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert A.dtype == np.int32 or A.dtype == np.float32, 'Floor division is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert B.dtype == np.int32 or B.dtype == np.float32, 'Floor division is only permitted using types: "Int", "Float"'
    
    return (A // B).astype(np.int32)


def modulo(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert A.dtype == np.int32 or A.dtype == np.float32, 'Modulo is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert B.dtype == np.int32 or B.dtype == np.float32, 'Modulo is only permitted using types: "Int", "Float"'
    
    if A.dtype != B.dtype:
        return (A % B).astype(np.float32)

    return A % B


def not_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    tile_offset = get_direction_offset(direction % 8)
    result = evaluate(x + tile_offset[0], y + tile_offset[1], world)
    assert result.dtype == bool, 'Not is only permitted using types: "Bool"'

    return np.invert(result)


def equal_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert A is not None, 'Equal attempted using invalid input'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert B is not None, 'Equal attempted using invalid input'

    return (A == B).astype(bool)


def greater_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert A.dtype == np.int32 or A.dtype == np.float32, 'Greater than is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert B.dtype == np.int32 or B.dtype == np.float32, 'Greater than is only permitted using types: "Int", "Float"'
    
    if hasattr(A, '__len__') and hasattr(B, '__len__'):
        assert (len(A) == 1 or len(B) == 1 or len(A) == len(B)), \
            'Greater than not permitted between lists of different shape' 

    return (A > B).astype(bool)


def lesser_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert A.dtype == np.int32 or A.dtype == np.float32, 'Lesser than is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert B.dtype == np.int32 or B.dtype == np.float32, 'Lesser than is only permitted using types: "Int", "Float"'
    
    if hasattr(A, '__len__') and hasattr(B, '__len__'):
        assert (len(A) == 1 or len(B) == 1 or len(A) == len(B)), \
            'Lesser than not permitted between lists of different shape' 

    return (A < B).astype(bool)


def read_array(x, y, world):
    direction, _ = read_tile(x, y, world)
    type_offset = get_direction_offset(direction)
    length_offset = get_direction_offset((direction+4) % 8)

    _, array_type = read_tile(x + type_offset[0], y + type_offset[1], world)
    array_length = evaluate(x + length_offset[0], y + length_offset[1], world)
    assert isinstance(array_length, np.int32), 'Array length must be of type: "Int"'
    assert array_type in [11, 12, 13, 14], 'Array can only be type: "Int", "Float", "Char", "Pointer"'

    array_offset = np.sum((get_direction_offset(np.array([direction, (direction+7)%8]))), axis=0)
    assert 0 <= x + (array_offset * array_length)[0] < world.shape[0], 'Array X coordinate is out of range'
    assert 0 <= y + (array_offset * array_length)[1] < world.shape[1], 'Array Y coordinate is out of range'
    
    data_locations = (array_offset[None, :] * np.arange(1, array_length+1)[:, None])
    data_locations[:, 0] += x
    data_locations[:, 1] += y

    assert np.all(world[data_locations[:, 0], data_locations[:, 1]] == world[x + type_offset[0], y + type_offset[1]])
    
    if array_type == 11:
        return read_int(data_locations[:, 0], data_locations[:, 1], world)

    if array_type == 12:
        return read_float(data_locations[:, 0], data_locations[:, 1], world)

    if array_type == 13:
        return read_char(data_locations[:, 0], data_locations[:, 1], world)

    if array_type == 14:
        pointer_list = []
        for array_x, array_y in data_locations:
            pointer_list.append(evaluate(array_x, array_y, world))
        
        return pointer_list


def write_array(x, y, world, value):
    pass


def read_array_index(x, y, world):
    direction, _ = read_tile(x, y, world)
    index_offset = get_direction_offset((direction+6) % 8)
    array_offset = get_direction_offset((direction+2) % 8)

    index = evaluate(x + index_offset[0], y + index_offset[1], world)
    array = evaluate(x + array_offset[0], y + array_offset[1], world)

    assert hasattr(array, '__len__'), "Attempted to access variable that was not Array"
    assert index.dtype == bool or index.dtype == np.int32, \
        'Array indexes can only be of type: "Int", "Bool"'

    if index.dtype == bool:
        assert hasattr(index, '__len__') and len(index) == len(array), \
            "Array Bool masks must be of same length as Array"
    
    if index.dtype == np.int32:
        assert np.min(index) >= 0 and np.max(index) < len(array), \
            "Attempted to access index out of range in Array"

    return array[index]


def write_array_index(x, y, world, value):
    pass


def read_ptr(x, y, world):
    direction, _ = read_tile(x, y, world)
    x_tile_offset = get_direction_offset((direction+6) % 8)
    y_tile_offset = get_direction_offset((direction+2) % 8)
    
    ptr_x = evaluate(x + x_tile_offset[0], y + x_tile_offset[1], world)
    assert isinstance(ptr_x, np.int32), 'Pointer coordinates must be of type: "Int"'
    assert 0 <= ptr_x < world.shape[0], 'Pointer X coordinate is out of range'

    ptr_y = evaluate(x + y_tile_offset[0], y + y_tile_offset[1], world)
    assert isinstance(ptr_y, np.int32), 'Pointer coordinates must be of type: "Int"'
    assert 0 <= ptr_y < world.shape[1], 'Pointer Y coordinate is out of range'

    return evaluate(ptr_x, ptr_y, world)


def write_ptr(x, y, world, value):
    direction, _ = read_tile(x, y, world)
    x_tile_offset = get_direction_offset((direction+6) % 8)
    y_tile_offset = get_direction_offset((direction+2) % 8)
    
    ptr_x = evaluate(x + x_tile_offset[0], y + x_tile_offset[1], world)
    assert isinstance(ptr_x, np.int32), 'Pointer coordinates must be of type: "Int"'
    assert 0 <= ptr_x < world.shape[0], 'Pointer X coordinate is out of range'

    ptr_y = evaluate(x + y_tile_offset[0], y + y_tile_offset[1], world)
    assert isinstance(ptr_y, np.int32), 'Pointer coordinates must be of type: "Int"'
    assert 0 <= ptr_y < world.shape[1], 'Pointer Y coordinate is out of range'

    return write(ptr_x, ptr_y, world, value)


def print_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    tile_offset = get_direction_offset(direction % 8)
    result = evaluate(x + tile_offset[0], y + tile_offset[1], world)
    assert result is not None, 'Print attempted using invalid input'
    print(result)
    return result


def input_tile(x, y, world):
    pass


def write_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    target_offset = get_direction_offset((direction+6) % 8)
    destination_offset = get_direction_offset((direction+2) % 8)
    
    target = evaluate(x + target_offset[0], y + target_offset[1], world)
    destination = evaluate(x + destination_offset[0], y + destination_offset[1], world)
    
    if hasattr(target, '__len__') and hasattr(destination, '__len__'):
        assert (len(target) == 1 or len(destination) == 1 or len(target) == len(destination)), \
            'Write not permitted between target and destination Arrays are different shape' 

    assert target.dtype == destination.dtype, "Target and destination type must match"

    write(x + destination_offset[0], y + destination_offset[1], world, target)


def null_tile(x, y, world):
    raise IndexError('Attempted to evaluate non-program memory at coordinates: {}-{}'.format(x, y))

    

