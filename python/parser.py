#!/usr/bin/env python
__doc__ = """

DataSpecParser class.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import ConfigParser

from vector import Vec3d

class Parser(object):
    """
    Parser class.
    """

    def __init__(self, dspec_path, net_spec, params, auto_mask=True):
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
        self._config   = config
        self.net_spec  = net_spec
        self.params    = params
        self.auto_mask = auto_mask

        # Set dataset-specific params.
        self.dparams = self._parse_dataset_params(config)


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
            # for name, data in self._config.items(section):
            for name in self.net_spec.keys():
                if self._config.has_option(section, name):
                    data = self._config.get(section, name)
                    config.set(section, name, data)
                    self.parse_data(config, name, data, dataset_id)
        else:
            raise RuntimeError('dataset section does not exist.')

        # Treat special case of affinity.
        self._treat_affinity(config)
        # Treat border.
        self._treat_border(config)

        # Dataset-specific params.
        dparams = dict()
        dparams['dataset_id'] = dataset_id
        for k, v in self.dparams.iteritems():
            dparams[k] = v[dataset_id]

        return config, dparams

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
        if self.auto_mask and 'label' in data:
            self._add_mask(config, name, data, idx)

    ####################################################################
    ## Private Helper Methods
    ####################################################################

    def _parse_dataset_params(self, config):
        """
        TODO(kisuk): Documentation.
        """
        ret = dict()
        if config.has_section('params'):
            opts = config.options('params')
            for opt in opts:
                ret[opt] = eval(config.get('params',opt))
        return ret

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
            config.set(mask, option, value)

        # Add default mask filler.
        # If mask is not built from file, build it from [shape] and [filler]
        # options. [shape] option should be added later dynamically.
        if not config.has_option(mask, 'file'):
            config.set(mask, 'shape', '(z,y,x)')  # Place holder
            config.set(mask, 'filler', "{'type':'one'}")

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
        if self._has_affinity(config):
            for _, data in config.items('dataset'):
                assert config.has_section(data)
                assert config.has_option(data, 'fov')
                # Increment FoV by 1.
                fov = config.get(data, 'fov')
                fov = tuple(x+1 for x in fov)
                config.set(data, 'fov', fov)

    def _treat_border(self, config):
        """
        TODO(kisuk): Documentation.
        """
        border_func = self.params.get('border', None)
        if border_func is not None:
            if border_func['type'] is 'mirror_border':
                for _, data in config.items('dataset'):
                    # Apply only to images.
                    if not 'image' in data:
                        continue
                    assert config.has_section(data)
                    assert config.has_option(data, 'offset')
                    offset = config.get(data, 'offset')
                    # Append border mirroring to the preprocessing list.
                    if config.has_option(data, 'preprocess'):
                        pp = config.get(data, 'Preprocess') + '\n'
                    else:
                        pp = ''
                    pp += str(border_func)
                    config.set(data, 'preprocess', pp)
                    # Update offset.
                    fov = border_func['fov']
                    offset = Vec3d(offset) - Vec3d(fov)/2
                    config.set(data, 'offset', tuple(offset))
            else:
                msg = 'unknown border mode [%s].' % border_func['type']
                raise RuntimeError(msg)


if __name__ == "__main__":

    # Data spec path.
    dspec_path = 'test_spec/zfish.spec'

    # FoV of the net.
    fov = (9,109,109)

    # For training.
    net_spec = dict(input=(18,208,208), label=(10,100,100))
    params = dict()
    params['border']  = dict(type='mirror_border', fov=fov)
    params['augment'] = [dict(type='flip')]
    params['drange']  = [0]

    # For inference.
    # net_spec = dict(input=(18,208,208))
    # params = dict()
    # params['border']  = dict(type='mirror_border', fov=fov)
    # params['drange']  = [0]

    # Parser
    p = Parser(dspec_path, net_spec, params)
    config, dparams = p.parse_dataset(0)
    assert dparams['dataset_id']==0
    print dparams
    f = open('zfish_dataset0.spec', 'w')
    config.write(f)
    f.close()
