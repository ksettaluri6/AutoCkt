import re
import copy
import os
from jinja2 import Environment, FileSystemLoader
import os
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
import yaml
import importlib
import random
import numpy as np
from bag_deep_ckt.eval_engines.util.core import IDEncoder, Design
from bag_deep_ckt.eval_engines.spectre.parser import SpectreParser
import IPython

debug = False 

def get_config_info():
    # TODO
    config_info = dict()
    base_tmp_dir = os.environ.get('BASE_TMP_DIR', None)
    if not base_tmp_dir:
        raise EnvironmentError('BASE_TMP_DIR is not set in environment variables')
    else:
        config_info['BASE_TMP_DIR'] = base_tmp_dir

    return config_info

class SpectreWrapper(object):

    def __init__(self, tb_dict):
        """
        This Wrapper handles one netlist at a time, meaning that if there are multiple test benches
        for characterizations multiple instances of this class needs to be created

        :param netlist_loc: the template netlist used for circuit simulation
        """

        # suppose we have a config_info = {'section':'model_lib'}
        # config_info also contains BASE_TMP_DIR (location for storing simulation netlist/results)
        # implement get_config_info() later

        netlist_loc = tb_dict['netlist_template']
        if not os.path.isabs(netlist_loc):
            netlist_loc = os.path.abspath(netlist_loc)
        pp_module = importlib.import_module(tb_dict['tb_module'])
        pp_class = getattr(pp_module, tb_dict['tb_class'])
        self.post_process = getattr(pp_class, tb_dict['post_process_function'])
        self.tb_params = tb_dict['tb_params']

        self.config_info = get_config_info()

        self.root_dir = self.config_info['BASE_TMP_DIR']
        self.num_process = self.config_info.get('NUM_PROCESS', 1)

        _, dsn_netlist_fname = os.path.split(netlist_loc)
        self.base_design_name = os.path.splitext(dsn_netlist_fname)[0] + str(random.randint(0,10000))
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        os.makedirs(self.gen_dir, exist_ok=True)

        file_loader = FileSystemLoader(os.path.dirname(netlist_loc))
        self.jinja_env = Environment(loader=file_loader)
        self.template = self.jinja_env.get_template(dsn_netlist_fname)

    def _get_design_name(self, state):
        """
        Creates a unique identifier fname based on the state
        :param state:
        :return:
        """
        fname = self.base_design_name
        for value in state.values():
            fname += "_" + str(value)
        return fname

    def _create_design(self, state, new_fname):
        output = self.template.render(**state)
        design_folder = os.path.join(self.gen_dir, new_fname)
        os.makedirs(design_folder, exist_ok=True)
        fpath = os.path.join(design_folder, new_fname + '.scs')
        with open(fpath, 'w') as f:
            f.write(output)
            f.close()
        return design_folder, fpath

    def _simulate(self, fpath):
        command = ['spectre', '%s'%fpath, '-format', 'psfbin' ,'> /dev/null 2>&1']
        log_file = os.path.join(os.path.dirname(fpath), 'log.txt')
        err_file = os.path.join(os.path.dirname(fpath), 'err_log.txt')

        with open(log_file, 'w') as file1, open(err_file,'w') as file2:
          exit_code = subprocess.call(command, cwd=os.path.dirname(fpath), stdout=file1, stderr=file2)

        info = 0
        if debug:
            print(command)
            print(fpath)
        if (exit_code % 256):
            info = 1 # this means an error has occurred

        return info


    def _create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        if debug:
            print('state', state)
            print('verbose', verbose)
        if dsn_name == None:
            dsn_name = self._get_design_name(state)
        else:
            dsn_name = str(dsn_name)
        if verbose:
            print(dsn_name)
        #add additional width metrics to 45nm netlist
        if "45nm" in dsn_name:
          wdict = {}
          for each_param in state:
            if not("cc" in each_param):
              wdict['w'+each_param] = state[each_param]*654e-9
          state.update(wdict)

        design_folder, fpath = self._create_design(state, dsn_name)
        info = self._simulate(fpath)
        results = self._parse_result(design_folder)
        if self.post_process:
            specs = self.post_process(results, self.tb_params)
            return state, specs, info
        specs = results
        return state, specs, info

    def _parse_result(self, design_folder):
        _, folder_name = os.path.split(design_folder)
        raw_folder = os.path.join(design_folder, '{}.raw'.format(folder_name))
        res = SpectreParser.parse(raw_folder)
       
        ##print(os.system("ls -l " + os.path.join("/proc",str(os.getpid()),"fd")))
        for fd in os.listdir(os.path.join("/proc", str(os.getpid()), "fd")):
            if int(fd) == 16:
              os.close(int(fd))

        #onlyfiles = [f for f in os.listdir(raw_folder) if os.path.isfile(os.path.join(raw_folder, f))]
        #for file in os.listdir(raw_folder):
        #  os.remove(os.path.join(raw_folder, file))
        return res

    def run(self, states, design_names=None, verbose=False):
        # TODO: Use asyncio to instantiate multiple jobs for running parallel sims
        """

        :param states:
        :param design_names: if None default design name will be used, otherwise the given design name will be used
        :param verbose: If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value), info: int)]
        """
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name)in zip(states, design_names)]
        specs = pool.starmap(self._create_design_and_simulate, arg_list)
        pool.close()
        return specs


class EvaluationEngine(object):

    def __init__(self, yaml_fname):

        self.design_specs_fname = yaml_fname
        with open(yaml_fname, 'r') as f:
            self.ver_specs = yaml.load(f)

        self.spec_range = self.ver_specs['spec_range']
        # params are interfaced using the index instead of the actual value
        params = self.ver_specs['params']

        self.params_vec = {}
        self.search_space_size = 1
        for key, value in params.items():
            self.params_vec[key] = np.arange(value[0], value[1], value[2]).tolist()
            self.search_space_size = self.search_space_size * len(self.params_vec[key])

        # minimum and maximum of each parameter
        # params_vec contains the acutal numbers but min and max should be the indices
        self.params_min = [0]*len(self.params_vec)
        self.params_max = []
        for val in self.params_vec.values():
            self.params_max.append(len(val)-1)

        self.id_encoder = IDEncoder(self.params_vec)
        self.measurement_specs = self.ver_specs['measurement']
        tbs = self.measurement_specs['testbenches']
        self.netlist_module_dict = {}
        for tb_kw, tb_val in tbs.items():
            self.netlist_module_dict[tb_kw] = SpectreWrapper(tb_val)


    @property
    def num_params(self):
        return len(self.params_vec)

    def generate_data_set(self, n=1, debug=False, parallel_config=None):
        """
        :param n:
        :return: a list of n Design objects with populated attributes (i.e. cost, specs, id)
        """
        valid_designs, tried_designs = [], []
        nvalid_designs = 0 

        useless_iter_count = 0
        while len(valid_designs) < n:
            design = {}
            for key, vec in self.params_vec.items():
                rand_idx = random.randrange(len(vec))
                design[key] = rand_idx
            design = Design(self.spec_range, self.id_encoder, list(design.values()))
            if design in tried_designs:
                if (useless_iter_count > n * 5):
                    raise ValueError("Random selection of a fraction of search space did not "
                                     "result in {} number of valid designs".format(n))
                useless_iter_count += 1
                continue
            design_result = self.evaluate([design], debug=debug)[0]
            if design_result['valid']:
                design.cost = design_result['cost']
                for key in design.specs.keys():
                    design.specs[key] = design_result[key]
                valid_designs.append(design)
            else:
                nvalid_designs += 1
            tried_designs.append(design)
            print(len(valid_designs))
            print("not valid designs:" + str(nvalid_designs))
        return valid_designs[:n]

    def evaluate(self, design_list, debug=False, parallel_config=None):
        """
        serial implementation of evaluate (not parallel)
        :param design_list:
        :return: a list of processed_results that the algorithm cares about, keywords should include
        cost and spec keywords with one scalar number as the value for each
        """
        results = []
        if len(design_list) > 1:
          for design in design_list:
              try:
                  result = self._evaluate(design, parallel_config=parallel_config)
                  #result['valid'] = True
              except Exception as e:
                  if debug:
                      raise e
                  result = {'valid': False}
                  print(getattr(e, 'message', str(e)))

              results.append(result)
        else:
          try:
            netlist_name, netlist_module = list(self.netlist_module_dict.items())[0]
            result = netlist_module._create_design_and_simulate(design_list[0])
          except Exception as e:
            if debug:
              raise e
            result = {'valid': False}
            print(getattr(e, 'message', str(e)))
          results.append(result)
        return results

    def _evaluate(self, design, parallel_config):
        state_dict = dict()
        for i, key in enumerate(self.params_vec.keys()):
            state_dict[key] = self.params_vec[key][design[i]]
        state = [state_dict]
        dsn_names = [design.id]
        print(state_dict)
        results = {}
        for netlist_name, netlist_module in self.netlist_module_dict.items():
            results[netlist_name] = netlist_module.create_design_and_simulate(state, dsn_names)

        specs_dict = self.get_specs(results, self.measurement_specs['meas_params'])
        print(specs_dict)
        specs_dict['cost'] = self.cost_fun(specs_dict)
        return specs_dict

    def find_worst(self, spec_nums, spec_kwrd, ret_penalty=False):
        if not hasattr(spec_nums, '__iter__'):
            spec_nums = [spec_nums]

        penalties = self.compute_penalty(spec_nums, spec_kwrd)
        worst_penalty = max(penalties)
        worst_idx = penalties.index(worst_penalty)
        if ret_penalty:
            return spec_nums[worst_idx], worst_penalty
        else:
            return spec_nums[worst_idx]

    def cost_fun(self, specs_dict):
        """
        :param design: a list containing relative indices according to yaml file
        :param verbose:
        :return:
        """
        cost = 0
        for spec in self.spec_range.keys():
            penalty = self.compute_penalty(specs_dict[spec], spec)[0]
            cost += penalty

        return cost

    def get_specs(self, results, params):
        """
        converts results from different tbs to a single dictionary summarizing everything
        :param results:
        :return:
        """
        raise NotImplementedError

    def compute_penalty(self, spec_nums, spec_kwrd):
        # use self.spec_range[spec_kwrd] to get the min, max, and weight
        raise NotImplementedError

def main():
  #testing the cs amp functionality with Jinja2
  dsn_netlist = 'bag_deep_ckt/eval_engines/spectre/netlist_templates/two_stage_opamp_16nm.scs'
  opamp_env = SpectreWrapper(tb_dict=dsn_netlist)

  w = 654e-9
  state = {"tail1":2, "tail2":4, "tailcm":0.5, "in":1000, "ref":1.0, "diode1":1.0, "ngm1":2.0, "diode2":2.0, "ngm2":2.0, "rfb":100, "cfb":10.0e-15}
  opamp_env._create_design_and_simulate(state)

if __name__=="__main__":
  main()
