import math
import time

from eval_engines.bag.bagEvalEngine import BagEvalEngine
from bag.io import read_yaml
from bag_deep_ckt.util import *
import pdb

class OpampEvaluationEngine(BagEvalEngine):
    def __init__(self, design_specs_fname):
        BagEvalEngine.__init__(self, design_specs_fname)
        self.ver_specs['measurements'][0]['find_cfb'] = False

    def impose_constraints(self, design_dict):
        design_dict['layout_params/seg_dict/tail1'] = design_dict['layout_params/seg_dict/in']
        return design_dict

    def process_results(self, results):
        # results = [{'funity': [blah, blah], 'pm': [blah, blah], 'gain': [blah, blah], 'corners', 'bw' }]
        # processed_results = [{'cost': blah, 'funity': blah, 'pm': blah}]
        processed_results = []
        for result in results:
            processed_result = {'valid': True}
            cost = 0
            if isinstance(result, Dict):
                result = result['opamp_ac']
                # just a couple of exception for definition of phase margin and gain in wierd cases
                # fow example when gain is less than unity, or when phase or gain are nan
                # in ac core pm is returned as nan if pm is larger than 180
                for i in range(len(result['funity'])):
                    if result['gain'][i] < 1.0:
                        result['pm'][i] = -180
                    if math.isnan(result['funity'][i]):
                        result['funity'][i] = 0
                    if math.isnan(result['pm'][i]):
                        result['pm'][i] = -180

                for spec in self.spec_range.keys():
                    processed_result[spec] = self.find_worst(result[spec], spec)
                    penalty = self.compute_penalty(processed_result[spec], spec)[0]
                    cost += penalty

                processed_result['cost'] = cost
            else:
                processed_result['valid'] = False

            processed_results.append(processed_result)

        return processed_results

    def find_worst(self, spec_nums, spec_kwrd):
        if type(spec_nums) is not list:
            spec_nums = [spec_nums]

        spec_min, spec_max, _ = self.spec_range[spec_kwrd]
        if spec_min is not None:
            return min(spec_nums)
        if spec_max is not None:
            return max(spec_nums)

    def compute_penalty(self, spec_nums, spec_kwrd):
        if type(spec_nums) is not list:
            spec_nums = [spec_nums]
        penalties = []
        for spec_num in spec_nums:
            penalty = 0
            spec_min, spec_max, w = self.spec_range[spec_kwrd]
            if spec_max is not None:
                if spec_num > spec_max:
                    # penalty += abs(spec_num / spec_max - 1.0)
                    penalty += w*abs((spec_num - spec_max) / (spec_num + spec_max))
            if spec_min is not None:
                if spec_num < spec_min:
                    # penalty += abs(spec_num / spec_min - 1.0)
                    penalty += w*abs((spec_num - spec_min) / (spec_num + spec_min))
            penalties.append(penalty)
        return penalties

    # def evaluate(self, design_list):
    #     # type: (List[Design]) -> List
    #
    #     swp_spec_file_list = []
    #     template = deepcopy(self.ver_specs)
    #     sweep_params_update = deepcopy(self.ver_specs['sweep_params'])
    #     # del template['sweep_params']['swp_spec_file']
    #     for dsn_num, design in enumerate(design_list):
    #         # 1. translate each list to a dict with layout_params and measurement_params indication
    #         # 2. write those dictionaries in the corresponding param.yaml and update self.ver_specs
    #         specs = deepcopy(template)
    #         layout_update, measurement_update = {}, {}
    #         layout_update['seg_dict'] = {}
    #         for value_idx, key in zip(design, self.params.keys()):
    #             if key in self.param_choices_layout.keys():
    #                 layout_update['seg_dict'][key] = self.params_vec[key][value_idx]
    #             elif key in self.param_choices_measurement.keys():
    #                 measurement_update[key] = self.params_vec[key][value_idx]
    #
    #         # imposing the constraint of layout generator
    #         layout_update['seg_dict']['tail1'] = layout_update['seg_dict']['in']
    #
    #         specs['layout_params'].update(layout_update)
    #         specs['measurements'][0].update(measurement_update)
    #         specs['sweep_params']['swp_spec_file'] = ['params'+str(dsn_num)]
    #
    #         swp_spec_file_list.append('params'+str(dsn_num))
    #         fname = os.path.join(self.swp_spec_dir, 'params'+str(dsn_num)+'.yaml')
    #         with open_file(fname, 'w') as f:
    #             yaml.dump(specs, f)
    #
    #     sweep_params_update['swp_spec_file'] = swp_spec_file_list
    #     self.ver_specs['sweep_params'].update(sweep_params_update)
    #     results = self.generate_and_sim()
    #     return self.process_results(results)

def main():

    eval_core = OpampEvaluationEngine(design_specs_fname='/tools/projects/ksettaluri6/BAG2_TSMC16FFC_tutorial/bag_deep_ckt/template_specs/opamp_two_stage_sim.yaml')
    content = read_yaml('/tools/projects/ksettaluri6/BAG2_TSMC16FFC_tutorial/bag_deep_ckt/template_specs/opamp_two_stage_sim.yaml')
    db_dir = content['database_dir']

    np.random.seed(10)
    random.seed(10)

    start = time.time()
    sample_designs = eval_core.generate_data_set(n=50, evaluate=True)
    print("time for simulating one instance: {}".format((time.time() - start)))

    pdb.set_trace()

    # import pickle
    # with open(db_dir+"/init_data.pickle", 'wb') as f:
    #     pickle.dump(sample_designs, f)
    #
    # with open(db_dir+"/init_data.pickle", 'rb') as f:
    #     data = pickle.load(f)
    #     a=1/0

    # designs = [[6, 6, 8, 8, 0, 3, 4, 8, 4, 31, 0]] # this design had a wierd phase behavior
    # designs = [[8, 6, 0, 8, 2, 0, 6, 7, 7, 21, 5]] # this design's funity for ff was nan
    # designs = [[8, 6, 0, 8, 2, 0, 6, 7, 7, 21, 5]]
    # results = eval_core.evaluate(designs)
    # fname = 'kourosh_test_opamp/swp_spec_files/param0.yaml'
    # data = read_yaml(fname)
    # pprint.pprint(data)

if __name__ == '__main__':
    main()
