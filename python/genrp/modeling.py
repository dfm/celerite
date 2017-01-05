# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from collections import OrderedDict

__all__ = ["Model", "ConstantModel"]


class Model(object):

    parameter_names = tuple()

    def __init__(self, *args):
        self.unfrozen_mask = np.ones(self.full_size, dtype=bool)
        self.dirty = True
        if len(args):
            self.parameter_vector = args

    def get_value(self, x):
        raise NotImplementedError("overloaded by subclasses")

    def __len__(self):
        return self.vector_size

    def __getitem__(self, name):
        return self.get_parameter(name)

    def __setitem__(self, name, value):
        return self.set_parameter(name, value)

    @property
    def full_size(self):
        return len(self.parameter_names)

    @property
    def vector_size(self):
        return self.unfrozen_mask.sum()

    @property
    def parameter_vector(self):
        return np.array([getattr(self, k) for k in self.parameter_names])

    @parameter_vector.setter
    def parameter_vector(self, v):
        for k, val in zip(self.parameter_names, v):
            setattr(self, k, val)
        self.dirty = True

    def get_parameter_dict(self, include_frozen=False):
        return OrderedDict(zip(
            self.get_parameter_names(include_frozen=include_frozen),
            self.get_parameter_vector(include_frozen=include_frozen),
        ))

    def get_parameter_names(self, include_frozen=False):
        if include_frozen:
            return self.parameter_names
        return tuple(p
                     for p, f in zip(self.parameter_names, self.unfrozen_mask)
                     if f)

    def get_parameter_vector(self, include_frozen=False):
        if include_frozen:
            return self.parameter_vector
        return self.parameter_vector[self.unfrozen_mask]

    def set_parameter_vector(self, vector, include_frozen=False):
        v = self.parameter_vector
        if include_frozen:
            v[:] = vector
        else:
            v[self.unfrozen_mask] = vector
        self.parameter_vector = v
        self.dirty = True

    def freeze_parameter(self, name):
        i = self.get_parameter_names(include_frozen=True).index(name)
        self.unfrozen_mask[i] = False

    def thaw_parameter(self, name):
        i = self.get_parameter_names(include_frozen=True).index(name)
        self.unfrozen_mask[i] = True

    def get_parameter(self, name):
        i = self.get_parameter_names(include_frozen=True).index(name)
        return self.get_parameter_vector(include_frozen=True)[i]

    def set_parameter(self, name, value):
        i = self.get_parameter_names(include_frozen=True).index(name)
        v = self.get_parameter_vector(include_frozen=True)
        v[i] = value
        self.set_parameter_vector(v, include_frozen=True)


class ConstantModel(Model):
    parameter_names = ("value")
    def get_value(self, x):
        return self.value + np.zeros_like(x)
