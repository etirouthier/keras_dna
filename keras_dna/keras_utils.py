#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:28:36 2020

@author: routhier
"""
import numpy as np
import sys


import six
try:
    import h5py
    HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
    h5py = None


if sys.version_info[0] == 3:
    import pickle
else:
    import cPickle as pickle

class H5Dict(object):
    """ A dict-like wrapper around h5py groups (or dicts).
    This allows us to have a single serialization logic
    for both pickling and saving to disk.
    Note: This is not intended to be a generic wrapper.
    There are lot of edge cases which have been hardcoded,
    and makes sense only in the context of model serialization/
    deserialization.
    # Arguments
        path: Either a string (path on disk), a Path, a dict, or a HDF5 Group.
        mode: File open mode (one of `{"a", "r", "w"}`).
    """

    def __init__(self, path, mode='a'):
        if isinstance(path, h5py.Group):
            self.data = path
            self._is_file = False
        elif isinstance(path, six.string_types) or _is_path_instance(path):
            self.data = h5py.File(path, mode=mode)
            self._is_file = True
        elif isinstance(path, dict):
            self.data = path
            self._is_file = False
            if mode == 'w':
                self.data.clear()
            # Flag to check if a dict is user defined data or a sub group:
            self.data['_is_group'] = True
        else:
            raise TypeError('Required Group, str, Path or dict. '
                            'Received: {}.'.format(type(path)))
        self.read_only = mode == 'r'

    @staticmethod
    def is_supported_type(path):
        """Check if `path` is of supported type for instantiating a `H5Dict`"""
        return (
            isinstance(path, h5py.Group) or
            isinstance(path, dict) or
            isinstance(path, six.string_types) or
            _is_path_instance(path)
        )

    def __setitem__(self, attr, val):
        if self.read_only:
            raise ValueError('Cannot set item in read-only mode.')
        is_np = type(val).__module__ == np.__name__
        if isinstance(self.data, dict):
            if isinstance(attr, bytes):
                attr = attr.decode('utf-8')
            if is_np:
                self.data[attr] = pickle.dumps(val)
                # We have to remember to unpickle in __getitem__
                self.data['_{}_pickled'.format(attr)] = True
            else:
                self.data[attr] = val
            return
        if isinstance(self.data, h5py.Group) and attr in self.data:
            raise KeyError('Cannot set attribute. '
                           'Group with name "{}" exists.'.format(attr))
        if is_np:
            dataset = self.data.create_dataset(attr, val.shape, dtype=val.dtype)
            if not val.shape:
                # scalar
                dataset[()] = val
            else:
                dataset[:] = val
        elif isinstance(val, (list, tuple)):
            # Check that no item in `data` is larger than `HDF5_OBJECT_HEADER_LIMIT`
            # because in that case even chunking the array would not make the saving
            # possible.
            bad_attributes = [x for x in val if len(x) > HDF5_OBJECT_HEADER_LIMIT]

            # Expecting this to never be true.
            if bad_attributes:
                raise RuntimeError('The following attributes cannot be saved to '
                                   'HDF5 file because they are larger than '
                                   '%d bytes: %s' % (HDF5_OBJECT_HEADER_LIMIT,
                                                     ', '.join(bad_attributes)))

            if (val and sys.version_info[0] == 3 and isinstance(
                    val[0], six.string_types)):
                # convert to bytes
                val = [x.encode('utf-8') for x in val]

            data_npy = np.asarray(val)

            num_chunks = 1
            chunked_data = np.array_split(data_npy, num_chunks)

            # This will never loop forever thanks to the test above.
            is_too_big = lambda x: x.nbytes > HDF5_OBJECT_HEADER_LIMIT
            while any(map(is_too_big, chunked_data)):
                num_chunks += 1
                chunked_data = np.array_split(data_npy, num_chunks)

            if num_chunks > 1:
                for chunk_id, chunk_data in enumerate(chunked_data):
                    self.data.attrs['%s%d' % (attr, chunk_id)] = chunk_data
            else:
                self.data.attrs[attr] = val
        else:
            self.data.attrs[attr] = val

    def __getitem__(self, attr):
        if isinstance(self.data, dict):
            if isinstance(attr, bytes):
                attr = attr.decode('utf-8')
            if attr in self.data:
                val = self.data[attr]
                if isinstance(val, dict) and val.get('_is_group'):
                    val = H5Dict(val)
                elif '_{}_pickled'.format(attr) in self.data:
                    val = pickle.loads(val)
                return val
            else:
                if self.read_only:
                    raise ValueError('Cannot create group in read-only mode.')
                val = {'_is_group': True}
                self.data[attr] = val
                return H5Dict(val)
        if attr in self.data.attrs:
            val = self.data.attrs[attr]
            if type(val).__module__ == np.__name__:
                if val.dtype.type == np.string_:
                    val = val.tolist()
        elif attr in self.data:
            val = self.data[attr]
            if isinstance(val, h5py.Dataset):
                val = np.asarray(val)
            else:
                val = H5Dict(val)
        else:
            # could be chunked
            chunk_attr = '%s%d' % (attr, 0)
            is_chunked = chunk_attr in self.data.attrs
            if is_chunked:
                val = []
                chunk_id = 0
                while chunk_attr in self.data.attrs:
                    chunk = self.data.attrs[chunk_attr]
                    val.extend([x.decode('utf8') for x in chunk])
                    chunk_id += 1
                    chunk_attr = '%s%d' % (attr, chunk_id)
            else:
                if self.read_only:
                    raise ValueError('Cannot create group in read-only mode.')
                val = H5Dict(self.data.create_group(attr))
        return val

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def iter(self):
        return iter(self.data)

    def __getattr__(self, attr):

        def wrapper(f):
            def h5wrapper(*args, **kwargs):
                out = f(*args, **kwargs)
                if isinstance(self.data, type(out)):
                    return H5Dict(out)
                else:
                    return out
            return h5wrapper

        return wrapper(getattr(self.data, attr))

    def close(self):
        if isinstance(self.data, h5py.Group):
            self.data.file.flush()
            if self._is_file:
                self.data.close()

    def update(self, *args):
        if isinstance(self.data, dict):
            self.data.update(*args)
        raise NotImplementedError

    def __contains__(self, key):
        if isinstance(self.data, dict):
            return key in self.data
        else:
            return (key in self.data) or (key in self.data.attrs)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

def _is_path_instance(path):
    # We can't use isinstance here because it would require
    # us to add pathlib2 to the Python 2 dependencies.
    class_name = type(path).__name__
    return class_name == 'PosixPath' or class_name == 'WindowsPath'
