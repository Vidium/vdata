# coding: utf-8
# Created on 21/10/2021 15:28
# Author : matteo

"""
Modified from https://github.com/DaanVanVugt/h5pickle/blob/master/h5pickle
"""

# ====================================================
# imports
import h5py

from typing import Any


# ====================================================
# code
def h5py_wrap_type(obj):
    """Produce our objects instead of h5py default objects"""
    if isinstance(obj, h5py.Dataset):
        return Dataset(obj.id)
    elif isinstance(obj, h5py.Group):
        return Group(obj.id)
    elif isinstance(obj, h5py.File):
        return File(obj.id)
    elif isinstance(obj, h5py.Datatype):
        return obj                          # Not supported for pickling yet. Haven't really thought about it
    else:
        return obj                          # Just return, since we want to wrap h5py.Group.get too


class PickleAbleH5PyObject(h5py.HLObject):
    """Save state required to pickle and unpickle h5py objects and groups.
    This class should not be used directly, but is here as a base for inheritance
    for Group and Dataset"""
    def __getstate__(self):
        """Save the current name and a reference to the root file object."""
        return {'name': self.name, 'file': self.file_info}

    def __setstate__(self,
                     state: dict[str, Any]):
        """File is reopened by pickle. Create a dataset and steal its identity"""
        self.__init__(state['file'][state['name']].id)
        self.file_info = state['file']

    def __getnewargs__(self):
        """Override the h5py getnewargs to skip its error message"""
        return ()


class Dataset(PickleAbleH5PyObject, h5py.Dataset):
    """Mix in our pickling class"""
    pass


class Group(PickleAbleH5PyObject, h5py.Group):
    """Overwrite group to allow pickling, and to create new groups and datasets
    of the right type (i.e. the ones defined in this module).
    """
    def __getitem__(self, name):
        obj = h5py_wrap_type(h5py.Group.__getitem__(self, name))
        # If it is a group or dataset copy the current file info in
        if isinstance(obj, Group) or isinstance(obj, Dataset):
            obj.file_info = self.file_info
        return obj


class File(PickleAbleH5PyObject, h5py.File):
    """A subclass of h5py.File that implements pickling.
    Pickling is done not with __{get,set}state__ but with __getnewargs_ex__
    which produces the arguments to supply to the __new__ method.
    """

    # noinspection PyMissingConstructor
    def __init__(self, *args, **kwargs):
        """We skip the init method, since it is called at object creation time
        by __new__. This is necessary to have both pickling and caching."""
        pass

    def __new__(cls, *args, **kwargs):
        """Create a new File object with the h5 open function."""
        self = super(h5py.File, cls).__new__(cls)
        h5py.File.__init__(self, *args, **kwargs)
        # Store args and kwargs for pickling
        self.init_args = args
        self.init_kwargs = kwargs

        return self

    def __getitem__(self, name):
        obj = h5py_wrap_type(h5py.Group.__getitem__(self, name))
        # If it is a group or dataset copy the current file info in
        if isinstance(obj, Group) or isinstance(obj, Dataset):
            obj.file_info = self
        return obj

    def __getstate__(self):
        pass

    def __getnewargs_ex__(self):
        kwargs = self.init_kwargs.copy()

        if len(self.init_args) > 1 and self.init_args[1] == 'w':
            return (self.init_args[0], 'r+', *self.init_args[2:]), kwargs

        return self.init_args, kwargs
