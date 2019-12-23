import os
from util import *
import abc
import yaml
from eval_engines.ngspice.TwoStageComplete import TwoStageMeasManager
import importlib

class NgspiceEvaluationCore(object):
    def __init__(self, design_specs_fname):
        self.design_specs_fname = design_specs_fname
        with open(design_specs_fname, 'r') as f:
            self.ver_specs = yaml.load(f)

        meas_module = importlib.import_module(self.ver_specs['measurement']['measurement_module'])
        meas_cls = getattr(meas_module, self.ver_specs['measurement']['measurement_class'])
        self.measMan = meas_cls(design_specs_fname)
        self.params = self.measMan.params
        self.spec_range = self.measMan.spec_range
        self.params_vec = self.measMan.params_vec
        # minimum and maximum of each parameter
        # params_vec contains the acutal numbers but min and max should be the indices
        self.params_min = [0]*len(self.params_vec)

        self.params_max = []
        for val in self.params_vec.values():
            self.params_max.append(len(val)-1)

        self.id_encoder = IDEncoder(self.params_vec)


    @property
    def num_params(self):
        return len(self.params_vec)

    def generate_data_set(self, n=1):
        """
        :param n:
        :param evaluate:
        :return: a list of n Design objects with populated attributes (i.e. cost, specs, id)
        """
        valid_designs, tried_designs = [], []

        useless_iter_count = 0
        while len(valid_designs) <= n:
            design = {}
            for key, vec in self.params_vec.items():
                rand_idx = random.randrange(len(vec))
                design[key] = rand_idx
            design = Design(self.spec_range, self.id_encoder, list(design.values()))
            if design in tried_designs:
                if (useless_iter_count > n * 5):
                    raise ValueError("Random selection of a fraction of search space did not result in {}"
                                     "number of valid designs".format(n))
                useless_iter_count += 1
                continue
            design_result = self.evaluate([design])[0]
            if design_result['valid']:
                design.cost = design_result['cost']
                for key in design.specs.keys():
                    design.specs[key] = design_result[key]
                valid_designs.append(design)
            tried_designs.append(design)
            print(len(valid_designs))

        return valid_designs[:n]

    def evaluate(self, design_list):
        """
        :param design_list:
        :return: a list of processed_results that the algorithm cares about, keywords should include
        cost and spec keywords with one scalar number as the value for each
        """
        results = []
        for design in design_list:
            try:
                result = self.measMan.evaluate(design)
                result['valid'] = True
            except Exception as e:
                result = {'valid': False}
                print(getattr(e, 'message', str(e)))
            results.append(result)
        return results

    def compute_penalty(self, spec_nums, spec_kwrd):
        return self.measMan.compute_penalty(spec_nums, spec_kwrd)


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


if __name__ == '__main__':
    yaml_file = './bag_deep_ckt/eval_engines/NGspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml'
    # yaml_file = './bag_deep_ckt/eval_engines/NGspice/ngspice_inputs/yaml_files/gb_none.yaml'
    eval_core = NgspiceEvaluationCore(yaml_file)


    with open(yaml_file, 'r') as f:
        content = yaml.load(f)

    db_dir = content['database_dir']
    os.makedirs(db_dir, exist_ok=True)

    np.random.seed(10)
    random.seed(10)

    start = time.time()
    data = eval_core.generate_data_set(n=10)
    print("time for simulating one instance: {}".format((time.time() - start)))

    import pdb
    pdb.set_trace()
    import pickle

    with open(db_dir + "/init_data.pickle", 'wb') as f:
        pickle.dump(data, f)

    with open(db_dir + "/init_data.pickle", 'rb') as f:
        data = pickle.load(f)
        data = sorted(data, key=lambda x:x.cost)
        a = 1 / 0