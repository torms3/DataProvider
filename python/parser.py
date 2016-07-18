#!/usr/bin/env python
__doc__ = """

DataSpecParser class.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import ConfigParser

class Parser(object):
    """
    Parser (working title) class.
    """

    def __init__(self, dspec_path, net_spec, params):
        """
        TODO(kisuk): Documentation.
        """
        # Construct a ConfigParser object.
        config = ConfigParser.ConfigParser()
        config.read(dspec_path)
        self._config = config

        # Net specification
        self.net_spec = net_spec

        # TODO(kisuk): Preprocess params.

    def parse_dataset(self, dataset_id):
        """
        TODO(kisuk): Documentation.

        Args:
            dataset_id:

        Returns:
            dataset:
        """
        config = ConfigParser.ConfigParser()

        # dataset section
        dataset = 'dataset%d' % dataset_id
        if self._config.has_section(dataset):
            config.add_section(dataset)
            for name, data in self._config.items(dataset):
                config.set(dataset, name, data)
                self.parse_data(config, name, data)
        else:
            err_msg = 'dataset section [%s] does not exist.' % dataset
            raise RuntimeError(err_msg)

        # Treat special case of affinity.
        # Increase FoV by 1, and then append crop to the transformation list.
        if self._has_affinity(config):
            self._treat_affinity(config)

        # Add border mirroring.
        if self.params['border_mode'] is 'mirror':
            self._treat_border(config)

        return config

    def parse_data(self, config, name, data):
        """
        TODO(kisuk): Documentation.
        """
        if self._config.has_section(data):
            config.add_section(data)
            for option, value in self._config.items(data):
                config.set(data, option, value)
        else:
            err_msg = 'data section [%s] does not exist.' % data
            raise RuntimeError(err_msg)

        # FoV
        fov = self.net_spec[name][-3:]
        config.set(data, 'fov', fov)

        # Add mask if data is label.
        if 'label' in data:
            data_id  = int(data.split('label')[-1])
            self._add_mask(config, data_id)

    def _add_mask(self, config, data_id):
        """
        TODO(kisuk): Documentation.
        """





def parse_data_spec(dspec_path, net_spec, drange, params):
    """
    Parse data spec and construct a ConfigParser object.

    Args:
        data_spec_path:
        net_spec:
        drange:
        params:

    Returns:
        config:
    """

    def instantiate_dataset(config, dataset_id):
        """
        Instantiate dataset template, if exists, with dataset_id.
        If dataset already exists, then do not instantiate.

        Returns:
            dataset: Section name for dataset.
        """
        template = 'dataset%d'
        dataset  = template % dataset_id

        if not config.has_section(dataset):
            if config.has_section(template):
                config.add_section(dataset)
                for name, data in config.items(template):
                    if '%d' in data:
                        data = data % dataset_id
                    config.set(dataset, name, data)
            else:
                err = 'dataset section [%s] does not exist.' % dataset
                raise RuntimeError(err)

        # Instantiate each data section.
        for name, data in config.items(dataset):
            instantiate_data(config, data)

        return dataset

    def instantiate_data(config, data):
        """
        Instantiate data template, if exists. If data section already exists
        (template specialization), then do not instantiate.
        """
        if 'image' in data:
            template = 'image%d'
            data_id  = int(data.split('image')[-1])
            is_label = False
        elif 'label' in data:
            template = 'label%d'
            data_id  = int(data.split('label')[-1])
            is_label = True
        else:
            raise RuntimeError('invalid data section [%s]' % data)

        if not config.has_section(data):
            if config.has_section(template):
                config.add_section(data)
                for name, value in config.items(template):
                    if '%d' in value:
                        value = value % data_id
                    config.set(data, name, value)
            else:
                raise RuntimeError('unknown data section [%s]' % data)

        # Instantiate mask if data is label.
        if is_label:
            instantiate_mask(config, data_id)

    def instantiate_mask(config, label_id):
        label = 'label%d' % label_id
        mask  = 'mask%d' % label_id

        # Add mask.
        config.add_section(mask)
        for name, value in config.items(label):
            if name is 'files':
                continue
            elif name is 'masks':
                name = 'files'
            elif name is 'transform':
                tf = eval(value.split('\n'))

            config.set(mask, name, value)

        # Add default mask filler.
        config.set(mask, 'filler', "{'type':'one'}")

    def is_affinity(config, data):
        """Check if data is affinity."""
        ret = False
        assert config.has_section(data)
        if 'label' in data:
            if config.has_option(data, 'transform'):
                tf = config.get(data, 'transform')
                if 'affinitize' in tf:
                    ret = True
        return ret

    def has_affinity(config, dataset):
        """Check if dataset has affinity data."""
        ret = False
        assert config.has_section(dataset)
        for _, data in config.items(dataset):
            if is_affinity(config, data):
                ret = True
                break
        return ret

    def treat_affinity(config, dataset):
        if has_affinity(config, dataset):
            for _, data in config.items(dataset):

    # Construct a ConfigParser
    config = ConfigParser.ConfigParser()
    config.read(dspec_path)

    for dataset_id in drange:
        # Instantiate template dataset.
        dataset = instantiate_dataset(config, dataset_id)
        # Treat affinity.

        # Add FoV.
        for name, data in config.items(dataset):
            fov = net_spec[name]
            config.set(data, 'fov', str(dim[-3:]))
        # Add border mirroring.

    return config