import numpy as np
from python_interpreter.utils import get_direction_offset, read_tile


def evaluate(x, y, world):
    evaluate_functions = {
        0:    arrow_tile,
        1:    addition_tile,
        2:    subtraction_tile,
        3:    multiplication_tile,
        4:    division_tile,
        5:    floor_division_tile,
        6:    modulo_tile,
        7:    not_tile,
        8:    equal_tile,
        9:    greater_tile,
        10:   lesser_tile,
        11:   int_tile,
        12:   float_tile,
    }

    _, instruction = read_tile(x, y, world)
    if instruction in evaluate_functions:
        return evaluate_functions[instruction](x, y, world)


def arrow_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    move_x, move_y = get_direction_offset(direction)
    return evaluate(x+move_x, y+move_y, world)


def addition_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert isinstance(A, (np.int32, np.float32)), 'Addition is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert isinstance(B, (np.int32, np.float32)), 'Addition is only permitted using types: "Int", "Float"'
    
    if type(A) != type(B):
        return np.float32(A + B)

    return A + B


def subtraction_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert isinstance(A, (np.int32, np.float32)), 'Subtraction is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert isinstance(B, (np.int32, np.float32)), 'Subtraction is only permitted using types: "Int", "Float"'
    
    if type(A) != type(B):
        return np.float32(A - B)

    return A - B


def multiplication_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert isinstance(A, (np.int32, np.float32)), 'Multiplication is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert isinstance(B, (np.int32, np.float32)), 'Multiplication is only permitted using types: "Int", "Float"'
    
    if type(A) != type(B):
        return np.float32(A * B)

    return A * B


def division_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert isinstance(A, (np.int32, np.float32)), 'Division is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert isinstance(B, (np.int32, np.float32)), 'Division is only permitted using types: "Int", "Float"'
    
    return np.float32(A / B)


def floor_division_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert isinstance(A, (np.int32, np.float32)), 'Floor division is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert isinstance(B, (np.int32, np.float32)), 'Floor division is only permitted using types: "Int", "Float"'
    
    return np.int32(A // B)


def modulo_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert isinstance(A, (np.int32, np.float32)), 'Modulo is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert isinstance(B, (np.int32, np.float32)), 'Modulo is only permitted using types: "Int", "Float"'
    
    if type(A) != type(B):
        return np.float32(A % B)

    return A % B


def not_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    tile_offset = get_direction_offset(direction % 8)
    result = evaluate(x + tile_offset[0], y + tile_offset[1], world)
    assert isinstance(result, bool), 'Not is only permitted using types: "Bool"'

    return not result


def equal_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert isinstance(A, (np.int32, np.float32, bool, list, str)), \
        'Equal is only permitted using types: "Int", "Float", "Bool", "List", "String"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert isinstance(A, (np.int32, np.float32, bool, list, str)), \
        'Equal is only permitted using types: "Int", "Float", "Bool", "List", "String"'
    
    # comparing between different types is always False
    if type(A) != type(B):
        return False
    
    # comparing between lists is False if any item is a different type than it's pair
    if isinstance(A, list):
        if len(A) != len(B):
            return False

        for el_A, el_B in zip(A, B):
            if type(el_A) != type(el_B):
                return False

    return bool(A == B)


def greater_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert isinstance(A, (np.int32, np.float32)), 'Greater than is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert isinstance(B, (np.int32, np.float32)), 'Greater than is only permitted using types: "Int", "Float"'
    
    return bool(A > B)


def lesser_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    
    A_tile_offset = get_direction_offset((direction+6) % 8)
    A = evaluate(x + A_tile_offset[0], y + A_tile_offset[1], world)
    assert isinstance(A, (np.int32, np.float32)), 'Greater than is only permitted using types: "Int", "Float"'

    B_tile_offset = get_direction_offset((direction+2) % 8)
    B = evaluate(x + B_tile_offset[0], y + B_tile_offset[1], world)
    assert isinstance(B, (np.int32, np.float32)), 'Greater than is only permitted using types: "Int", "Float"'
    
    return bool(A < B)


def int_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    data_offsets = np.array([x, y]) + get_direction_offset((direction + np.arange(0, 8, 2))%8)
    value_bytes = np.packbits(world[data_offsets[:, 0], data_offsets[:, 1]])
    value = np.frombuffer(value_bytes, dtype = np.int32)[0]
    return value


def float_tile(x, y, world):
    direction, _ = read_tile(x, y, world)
    data_offsets = np.array([x, y]) + get_direction_offset((direction + np.arange(0, 8, 2))%8)
    value_bytes = np.packbits(world[data_offsets[:, 0], data_offsets[:, 1]])
    value = np.frombuffer(value_bytes, dtype = np.float32)[0]
    return value