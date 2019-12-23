from eval_engines.spectre.core import EvaluationEngine
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt
import pdb

class CSMeasMan(EvaluationEngine):

    def __init__(self, yaml_fname):
        EvaluationEngine.__init__(self, yaml_fname)


    def get_specs(self, results_dict, params):
        specs_dict = dict()
        ac_dc = results_dict['ac_dc']
        for _, res, _ in ac_dc:
            specs_dict = res

        return specs_dict

    def compute_penalty(self, spec_nums, spec_kwrd):
        if type(spec_nums) is not list:
            spec_nums = [spec_nums]
        penalties = []
        for spec_num in spec_nums:
            penalty = 0
            spec_min, spec_max, w = self.spec_range[spec_kwrd]
            if spec_max is not None:
                if spec_num > spec_max:
                    penalty += w * abs(spec_num - spec_max) / abs(spec_num)
            if spec_min is not None:
                if spec_num < spec_min:
                    penalty += w * abs(spec_num - spec_min) / abs(spec_min)
            penalties.append(penalty)
        return penalties



class ACTB(object):

    @classmethod
    def process_ac(cls, results, params):
        ac_result = results['ac']
        dc_results = results['dcOpc']

        vout = ac_result['out']
        freq = ac_result['sweep_values']

        gain = cls.find_dc_gain(vout)
        bw = cls.find_bw(vout, freq)

        results = dict(
            gain=gain,
            bw=bw
        )
        return results

    @classmethod
    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    @classmethod
    def find_bw(self, vout, freq):
        gain = np.abs(vout)
        gain_3dB = gain[0] / np.sqrt(2)
        return self._get_best_crossing(freq, gain, gain_3dB)

    @classmethod
    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop)
        except ValueError:
            # avoid no solution
            if abs(fzero(xstart)) < abs(fzero(xstop)):
                return xstart
            return xstop

if __name__ == '__main__':

    yname = 'bag_deep_ckt/eval_engines/spectre/specs_test/common_source.yaml'
    eval_core = CSMeasMan(yname)

    designs = eval_core.generate_data_set(n=1)
    pdb.set_trace()