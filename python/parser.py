#!/usr/bin/env python
__doc__ = """

DataSpecParser class.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import ConfigParser

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
        Instantiate dataset template, if exists, with dataset_id. If dataset
        already exists (template specialization), then do not instantiate.

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
                raise RuntimeError('unknown dataset section [%s]' % dataset)

        # Instantiate each data section.
        for name, data in config.items(dataset):
            instantiate_data(config, data)

        return section

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