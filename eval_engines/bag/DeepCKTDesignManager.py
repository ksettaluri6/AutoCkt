from bag.simulation.core import DesignManager
from copy import deepcopy

from typing import Dict, Any, Tuple, Sequence

import importlib

from bag.io import read_yaml
from bag.layout import TemplateDB
from bag.concurrent.core import batch_async_task
import pdb
import pprint

class DeepCKTDesignManager(DesignManager):
    # Modifications:
    #   get_layout -> replaces the lay_params with the correct values
    #   create_designs -> doesn't create the template data base, rather
    #   it assumes it is already set, which happens from outside

    # TODO Come up with a better DeisgnManager
    # 1. It should associate schematic and layout together always
    # 2. Should not replace layout and dump the schematic in somewhere else
    # 3. make gen_wrapper=True if wrapper is given, if not make it false
    # 4. make it such that the circuit can be broken down to sub circuits and individually perform simulation on it

    def __init__(self, *args, **kwargs):
        self._temp_db = None
        DesignManager.__init__(self, *args, **kwargs)

    def set_tdb(self, tdb):
        self._temp_db = tdb

    def get_layout_params(self, val_list):
        # type: (Tuple[Any, ...]) -> Dict[str, Any]
        """Returns the layout dictionary from the given sweep parameter values. Overwritten to incorporate
        the discrete evaluation problem, instead of a sweep"""

        # sanity check
        assert len(self.swp_var_list) == 1 and 'swp_spec_file' in self.swp_var_list , \
            "when using DeepCKTDesignManager for replacing file name, just (swp_spec_file) should be part of the " \
            "sweep_params dictionary"

        lay_params = deepcopy(self.specs['layout_params'])
        yaml_fname = self.specs['root_dir']+'/gen_yamls/swp_spec_files/' + val_list[0] + '.yaml'
        print(yaml_fname)
        updated_top_specs = read_yaml(yaml_fname)
        new_lay_params = updated_top_specs['layout_params']
        lay_params.update(new_lay_params)
        return lay_params

    def create_designs(self, create_layout):
        # type: (bool) -> None
        """Create DUT schematics/layouts.
        """
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')
        ##########################################################
        ############## Change made here:
        ##########################################################
        if self._temp_db is None:
            self._temp_db = self.make_tdb()
        temp_db = self._temp_db

        # make layouts
        dsn_name_list, lay_params_list, combo_list_list = [], [], []
        for combo_list in self.get_combinations_iter():
            dsn_name = self.get_design_name(combo_list)
            lay_params = self.get_layout_params(combo_list)
            dsn_name_list.append(dsn_name)
            lay_params_list.append(lay_params)
            combo_list_list.append(combo_list)
        import time
        start = time.time()
        if create_layout:
            print('creating all layouts.')
            sch_params_list = self.create_dut_layouts(lay_params_list, dsn_name_list, temp_db, throw_errors=False,
                                                      log_file='./layout_errors.txt')
        else:
            print('schematic simulation, skipping layouts.')
            sch_params_list = [self.get_schematic_params(combo_list)
                               for combo_list in self.get_combinations_iter()]
        print("Layout Creation time:{}".format(time.time()-start))
        start = time.time()
        print('creating all schematics.')
        self.create_dut_schematics(sch_params_list, dsn_name_list, gen_wrappers=True, throw_errors=False)

        print("Schematic Creation time:{}".format(time.time()-start))
        print('design generation done.')

    def characterize_designs(self, generate=True, measure=True, load_from_file=False):
        # type: (bool, bool, bool) -> None
        """Sweep all designs and characterize them.

        Parameters
        ----------
        generate : bool
            If True, create schematic/layout and run LVS/RCX.
        measure : bool
            If True, run all measurements.
        load_from_file : bool
            If True, measurements will load existing simulation data
            instead of running simulations.
        """
        # import time
        # start = time.time()
        if generate:
            extract = self.specs['view_name'] != 'schematic'
            self.create_designs(extract)
        else:
            extract = False
        # print("design_creation_time = {}".format(time.time()-start))
        rcx_params = self.specs.get('rcx_params', None)
        impl_lib = self.specs['impl_lib']
        dsn_name_list = [self.get_design_name(combo_list)
                         for combo_list in self.get_combinations_iter()]

        coro_list = [self.main_task(impl_lib, dsn_name, rcx_params, extract=extract,
                                    measure=measure, load_from_file=load_from_file)
                     for dsn_name in dsn_name_list]

        results = batch_async_task(coro_list)
        # if results is not None:
        #     for val in results:
        #         if isinstance(val, Exception):
        #             raise val
        return results

    def create_dut_schematics(self, sch_params_list, cell_name_list, gen_wrappers=True, throw_errors=True):
        # type: (Sequence[Dict[str, Any]], Sequence[str], bool) -> None
        dut_lib = self.specs['dut_lib']
        dut_cell = self.specs['dut_cell']
        impl_lib = self.specs['impl_lib']
        wrapper_list = self.specs['dut_wrappers']

        inst_list, name_list = [], []
        for sch_params, cur_name in zip(sch_params_list, cell_name_list):
            try:
                dsn = self.prj.create_design_module(dut_lib, dut_cell)
                dsn.design(**sch_params)
                inst_list.append(dsn)
                name_list.append(cur_name)
                if gen_wrappers:
                    for wrapper_config in wrapper_list:
                        wrapper_name = wrapper_config['name']
                        wrapper_lib = wrapper_config['lib']
                        wrapper_cell = wrapper_config['cell']
                        wrapper_params = wrapper_config['params'].copy()
                        wrapper_params['dut_lib'] = impl_lib
                        wrapper_params['dut_cell'] = cur_name
                        dsn = self.prj.create_design_module(wrapper_lib, wrapper_cell)
                        dsn.design(**wrapper_params)
                        inst_list.append(dsn)
                        name_list.append(self.get_wrapper_name(cur_name, wrapper_name))
            except Exception as e:
                if throw_errors:
                    raise e

        if inst_list:
            self.prj.batch_schematic(impl_lib, inst_list, name_list=name_list)

    def create_dut_layouts(self, lay_params_list, cell_name_list, temp_db, throw_errors=True, log_file=None):
        # type: (Sequence[Dict[str, Any]], Sequence[str], TemplateDB) -> Sequence[Dict[str, Any]]
        """Create multiple layouts"""
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')

        assert len(lay_params_list) == len(cell_name_list), 'lay_param_list and cell_name_list length mismatch'

        cls_package = self.specs['layout_package']
        cls_name = self.specs['layout_class']

        lay_module = importlib.import_module(cls_package)
        temp_cls = getattr(lay_module, cls_name)

        temp_list, sch_params_list = [], []
        removing_indices = []
        if log_file:
            pprint.pprint("******** layout log file:", stream=open(log_file, 'w'))
        for i, (lay_params, cell_name) in enumerate(zip(lay_params_list, cell_name_list)):
            try:
                template = temp_db.new_template(params=lay_params, temp_cls=temp_cls)
                temp_list.append(template)
                sch_params_list.append(template.sch_params)
            except Exception as e:
                if throw_errors:
                    raise e
                else:
                    if log_file:
                        pprint.pprint("-"*50, stream=open(log_file, 'a'))
                        pprint.pprint(e, stream=open(log_file, 'a'))
                        pprint.pprint(lay_params, stream=open(log_file, 'a'))
                    removing_indices.append(i)

        for i in sorted(removing_indices, reverse=True):
            del cell_name_list[i]

        temp_db.batch_layout(self.prj, temp_list, cell_name_list)
        return sch_params_list
