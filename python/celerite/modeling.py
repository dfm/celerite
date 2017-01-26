# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from itertools import chain
from collections import OrderedDict

__all__ = ["Model", "ModelSet", "ConstantModel"]


class Model(object):

    parameter_names = tuple()

    def __init__(self, *args, **kwargs):
        self.unfrozen_mask = np.ones(self.full_size, dtype=bool)
        self.dirty = True

        # Pull out the recognized keyword arguments
        self.parameter_bounds = kwargs.pop(
            "bounds", [(None, None) for _ in range(self.full_size)])
        if len(self.parameter_bounds) != self.full_size:
            raise ValueError("the number of bounds must equal the number of "
                             "parameters")

        # Parameter values can be specified as arguments or keywords
        if len(args):
            if len(args) != self.full_size:
                raise ValueError("expected {0} arguments but got {1}"
                                 .format(self.full_size, len(args)))
            if len(kwargs):
                raise ValueError("parameters must be fully specified by "
                                 "arguments or keyword arguments, not both")
            self.parameter_vector = args

        else:
            # Loop over the kwargs and set the parameter values
            params = []
            for k in self.parameter_names:
                v = kwargs.pop(k, None)
                if v is None:
                    raise ValueError("missing parameter '{0}'".format(k))
                params.append(v)
            self.parameter_vector = params

            if len(kwargs):
                raise ValueError("unrecognized parameter(s) '{0}'"
                                 .format(list(kwargs.keys())))

    def get_value(self, x):
        raise NotImplementedError("overloaded by subclasses")

    def __len__(self):
        return self.vector_size

    def _get_name(self, name_or_index):
        try:
            int(name_or_index)
        except TypeError:
            return name_or_index
        return self.get_parameter_names()[int(name_or_index)]

    def __getitem__(self, name_or_index):
        return self.get_parameter(self._get_name(name_or_index))

    def __setitem__(self, name_or_index, value):
        return self.set_parameter(self._get_name(name_or_index), value)

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
        if len(v) != self.full_size:
            raise ValueError("dimension mismatch")
        for k, val in zip(self.parameter_names, v):
            setattr(self, k, float(val))
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

    def get_parameter_bounds(self, include_frozen=False):
        if include_frozen:
            return self.parameter_bounds
        return list(p
                    for p, f in zip(self.parameter_bounds, self.unfrozen_mask)
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

    def freeze_all_parameters(self):
        self.unfrozen_mask[:] = False

    def thaw_all_parameters(self):
        self.unfrozen_mask[:] = True

    def get_parameter(self, name):
        i = self.get_parameter_names(include_frozen=True).index(name)
        return self.get_parameter_vector(include_frozen=True)[i]

    def set_parameter(self, name, value):
        i = self.get_parameter_names(include_frozen=True).index(name)
        v = self.get_parameter_vector(include_frozen=True)
        v[i] = value
        self.set_parameter_vector(v, include_frozen=True)

    def log_prior(self):
        for p, b in zip(self.parameter_vector, self.parameter_bounds):
            if b[0] is not None and p < b[0]:
                return -np.inf
            if b[1] is not None and p > b[1]:
                return -np.inf
        return 0.0


class ModelSet(Model):

    def __init__(self, models):
        self.models = OrderedDict()
        for name, model in models:
            self.models[name] = model

    @property
    def dirty(self):
        return any(m.dirty for m in self.models.values())

    @dirty.setter
    def dirty(self, value):
        for m in self.models.values():
            m.dirty = value

    @property
    def full_size(self):
        return sum(m.full_size for m in self.models.values())

    @property
    def vector_size(self):
        return sum(m.vector_size for m in self.models.values())

    @property
    def unfrozen_mask(self):
        return np.concatenate([
            m.unfrozen_mask for m in self.models.values()
        ])

    @property
    def parameter_vector(self):
        return np.concatenate([
            m.parameter_vector for m in self.models.values()
        ])

    @parameter_vector.setter
    def parameter_vector(self, v):
        i = 0
        for m in self.models.values():
            l = m.full_size
            m.parameter_vector = v[i:i+l]
            i += l

    @property
    def parameter_names(self):
        return tuple(chain(*(
            map("{0}:{{0}}".format(name).format, m.parameter_names)
            for name, m in self.models.items()
        )))

    @property
    def parameter_bounds(self):
        return list(chain(*(
            m.parameter_bounds for m in self.models.values()
        )))

    def _apply_to_parameter(self, func, name, *args):
        comp = name.split(":")
        if comp[0] not in self.models:
            raise ValueError("unrecognized parameter '{0}'".format(name))
        return getattr(self.models[comp[0]], func)(":".join(comp[1:]), *args)

    def freeze_parameter(self, name):
        self._apply_to_parameter("freeze_parameter", name)

    def thaw_parameter(self, name):
        self._apply_to_parameter("thaw_parameter", name)

    def get_parameter(self, name):
        return self._apply_to_parameter("get_parameter", name)

    def set_parameter(self, name, value):
        self.dirty = True
        return self._apply_to_parameter("set_parameter", name, value)

    def log_prior(self):
        lp = 0.0
        for model in self.models.values():
            lp += model.log_prior()
            if not np.isfinite(lp):
                return -np.inf
        return lp


class ConstantModel(Model):
    parameter_names = ("value", )

    def get_value(self, x):
        return self.value + np.zeros_like(x)
