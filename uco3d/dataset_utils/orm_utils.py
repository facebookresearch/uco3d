# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This functionality requires SQLAlchemy 2.0 or later.

import math
import struct
from typing import Tuple

import numpy as np

from sqlalchemy import LargeBinary
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass
from sqlalchemy.types import TypeDecorator


# these produce policies to serialize structured types to blobs
def ArrayTypeFactory(shape=None):
    if shape is None:

        class VariableShapeNumpyArrayType(TypeDecorator):
            impl = LargeBinary

            def process_bind_param(self, value, dialect):
                if value is None:
                    return None

                ndim_bytes = np.int32(value.ndim).tobytes()
                shape_bytes = np.array(value.shape, dtype=np.int64).tobytes()
                value_bytes = value.astype(np.float32).tobytes()
                return ndim_bytes + shape_bytes + value_bytes

            def process_result_value(self, value, dialect):
                if value is None:
                    return None

                ndim = np.frombuffer(value[:4], dtype=np.int32)[0]
                value_start = 4 + 8 * ndim
                shape = np.frombuffer(value[4:value_start], dtype=np.int64)
                assert shape.shape == (ndim,)
                return np.frombuffer(value[value_start:], dtype=np.float32).reshape(
                    shape
                )

        return VariableShapeNumpyArrayType

    class NumpyArrayType(TypeDecorator):
        impl = LargeBinary

        def process_bind_param(self, value, dialect):
            if value is not None:
                if value.shape != shape:
                    raise ValueError(f"Passed an array of wrong shape: {value.shape}")
                return value.astype(np.float32).tobytes()
            return None

        def process_result_value(self, value, dialect):
            if value is not None:
                return np.frombuffer(value, dtype=np.float32).reshape(shape)
            return None

    return NumpyArrayType


def TupleTypeFactory(dtype=float, shape: Tuple[int, ...] = (2,)):
    format_symbol = {
        float: "f",  # float32
        int: "i",  # int32
    }[dtype]

    class TupleType(TypeDecorator):
        impl = LargeBinary
        _format = format_symbol * math.prod(shape)

        def process_bind_param(self, value, _):
            if value is None:
                return None

            if len(shape) > 1:
                value = np.array(value, dtype=dtype).reshape(-1)

            return struct.pack(TupleType._format, *value)

        def process_result_value(self, value, _):
            if value is None:
                return None

            loaded = struct.unpack(TupleType._format, value)
            if len(shape) > 1:
                loaded = _rec_totuple(
                    np.array(loaded, dtype=dtype).reshape(shape).tolist()
                )

            return loaded

    return TupleType


def _rec_totuple(t):
    if isinstance(t, list):
        return tuple(_rec_totuple(x) for x in t)

    return t


class Base(MappedAsDataclass, DeclarativeBase):
    """subclasses will be converted to dataclasses"""
