import numpy as np
from src_py.utils import read_tile_data, write_tile_data


def read_int(x, y, world):
    value_bytes = read_tile_data(x, y, world)
    value = np.frombuffer(value_bytes, dtype = np.int32).squeeze()[()]
    return value


def write_int(x, y, world, value):
    assert value.dtype==np.int32
    if hasattr(value, '__len__'):
        assert len(value) == 1, "Tried to write too many values to Int tile"

    write_tile_data(x, y, world, value)


def read_float(x, y, world):
    value_bytes = read_tile_data(x, y, world)
    value = np.frombuffer(value_bytes, dtype = np.float32).squeeze()[()]
    return value


def write_float(x, y, world, value):
    assert value.dtype==np.float32
    if hasattr(value, '__len__'):
        assert len(value) == 1, "Tried to write too many values to Float tile"

    write_tile_data(x, y, world, value)


def read_char(x, y, world):
    value_bytes = read_tile_data(x, y, world)
    string = (b'\xff\xfe\x00\x00' + value_bytes.tobytes()).decode("utf-32")
    return string


def write_char(x, y, world, value):
    assert isinstance(value, str)
    if hasattr(value, '__len__'):
        assert len(value) == 1, "Tried to write too many values to Char tile"

    value_data = value.encode('utf-32')
    value_int = np.frombuffer(value_data[4:], dtype=np.int32).squeeze()[()]
    write_tile_data(x, y, world, value_int)