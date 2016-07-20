#!/usr/bin/env python
__doc__ = """

DataSpecParser class.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import ConfigParser
from transform import label_transform
from vector import Vec3d

class Parser(object):
    """
    Parser class.
    """

    def __init__(self, dspec_path, net_spec, params):
        """Initialize a Parser object.

        Args:
            dspec_path: Data spec path.
            net_spec: Net spec, a dictionary containing layer-data name pairs.
            params: Parameter dictionary.
        """
        # Construct a ConfigParser object.
        config = ConfigParser.ConfigParser()
        config.read(dspec_path)

        # Set attributes.
        self._config  = config
        self.net_spec = net_spec
        self.params   = params

    def parse_dataset(self, dataset_id):
        """
        TODO(kisuk): Documentation.

        Args:
            dataset_id:

        Returns:
            dataset: ConfigParser object containing dataset info.
        """
        config = ConfigParser.ConfigParser()

        section = 'dataset'
        if self._config.has_section(section):
            config.add_section(section)
            for name, data in self._config.items(section):
                config.set(section, name, data)
                self.parse_data(config, name, data, dataset_id)
        else:
            raise RuntimeError('dataset section does not exist.')

        # Treat special case of affinity.
        self._treat_affinity(config)
        # Treat border mirroring.
        self._treat_border(config)

        return config

    def parse_data(self, config, name, data, idx):
        """
        TODO(kisuk): Documentation.

        Args:
            config:
            name:
            data:
            idx:
        """
        if self._config.has_section(data):
            config.add_section(data)
            for option, value in self._config.items(data):
                config.set(data, option, value)
        else:
            raise RuntimeError('data section [%s] does not exist.' % data)

        # File
        if config.has_option(data, 'file'):
            value = config.get(data, 'file')
            assert self._config.has_option('files', value)
            flist = self._config.get('files', value).split('\n')
            config.set(data, 'file', flist[idx])

        # FoV
        fov = self.net_spec[name][-3:]
        config.set(data, 'fov', fov)

        # Offset
        if not config.has_option(data, 'offset'):
            config.set(data, 'offset', (0,0,0))

        # Add mask if data is label.
        if 'label' in data:
            self._add_mask(config, name, data, idx)

    ####################################################################
    ## Private Helper Methods
    ####################################################################

    def _add_mask(self, config, name, data, idx):
        """Add mask for label.

        Each label is supposed to have corresponding mask, an equal-sized
        volume with postive real values. During training, cost and gradient
        volumes are element-wise multiplied by mask.

        Args:
            config: ConfigParser object containing dataset info.
            name: Layer name for label.
            data: Section name for label data.
            idx: Dataset ID.
        """
        assert config.has_section(data)

        # Mask name is assumed to be label name with '_mask' postfix.
        name = name + '_mask'
        mask = data + '_mask'
        config.set('dataset', name, mask)

        # Add mask data.
        config.add_section(mask)
        for option, value in config.items(data):
            if option == 'file':
                continue
            elif option == 'mask':
                # Use [mask] option of label, if exists, to instantiate
                # [file] option of mask.
                assert self._config.has_option('files', value)
                flist  = self._config.get('files', value).split('\n')
                option = 'file'
                value  = flist[idx]
            elif option == 'transform':
                tf = [eval(x) for x in value.split('\n')]
                tf = self._treat_mask_transform(tf)
                value = '\n'.join([str(x) for x in tf])
            config.set(mask, option, value)

        # Add default mask filler.
        # If mask is not built from file, build it from [shape] and [filler]
        # options. [shape] option should be added later dynamically.
        if not config.has_option(mask, 'file'):
            config.set(mask, 'shape', '(z,y,x)')  # Place holder
            config.set(mask, 'filler', "{'type':'one'}")

    def _treat_mask_transform(self, tf):
        """
        Replace label transformation with mask transformation by adding
        [is_mask=True] optional argument.

        Args:
            tf: List of transformation.

        Returns:
            ret: List of transformation, with label transformations substituted
                 by corresponding mask transformations.
        """
        ret = []
        for t in tf:
            if t['type'] in label_transform:
                t['is_mask'] = True
            ret.append(t)
        return ret

    def _is_affinity(self, config, data):
        """Check if data is affinity."""
        ret = False
        if config.has_section(data):
            if config.has_option(data, 'transform'):
                tf = config.get(data, 'transform')
                if 'affinitize' in tf:
                    ret = True
        return ret

    def _has_affinity(self, config):
        """Check if dataset contains affinity data."""
        ret = False
        for _, data in config.items('dataset'):
            if self._is_affinity(config, data):
                ret = True
                break
        return ret

    def _treat_affinity(self, config):
        """
        TODO(kisuk): Documentation.
        """
        for _, data in config.items('dataset'):
            assert config.has_section(data)
            assert config.has_option(data, 'fov')
            # Increment FoV by 1.
            fov = config.get(data, 'fov')
            fov = tuple(x+1 for x in fov)
            config.set(data, 'fov', fov)
            # Append crop to the transformation list.
            if config.has_option(data, 'transform'):
                tf = config.get(data, 'transform') + '\n'
            else:
                tf = ''
            tf += "{'type':'crop','offset':(1,1,1)}"
            config.set(data, 'transform', tf)

    def _treat_border(self, config):
        """
        TODO(kisuk): Documentation.
        """
        if self.params['border_mode'] is 'mirror':
            for _, data in config.items('dataset'):
                # Apply only to images.
                if not 'image' in data:
                    continue
                assert config.has_section(data)
                assert config.has_option(data, 'fov')
                assert config.has_option(data, 'offset')
                fov = config.get(data, 'fov')
                off = config.get(data, 'offset')
                # Append border mirroring to the preprocessing list.
                if config.has_option(data, 'preprocess'):
                    pp = config.get(data, 'Preprocess') + '\n'
                else:
                    pp = ''
                pp += "{'type':'mirror_border','fov':(%d,%d,%d)}" % fov
                config.set(data, 'preprocess', pp)
                # Update offset.
                off = Vec3d(off) - Vec3d(fov)/2
                config.set(data, 'offset', tuple(off))