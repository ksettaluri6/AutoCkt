from bag_deep_ckt.eval_engines.spectre.core import EvaluationEngine
import numpy as np
import pdb
import IPython
import scipy.interpolate as interp
import scipy.optimize as sciopt
import matplotlib.pyplot as plt

class OpampMeasMan(EvaluationEngine):

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
        dc_results = results['dcOp']

        vout = ac_result['vout']
        freq = ac_result['sweep_values']

        gain = cls.find_dc_gain(vout)
        ugbw,valid = cls.find_ugbw(freq, vout)
        phm = cls.find_phm(freq, vout)

        results = dict(
            gain=gain,
            funity = ugbw,
            pm = phm,
            valid = valid
        )
        return results

    @classmethod
    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    @classmethod
    def find_ugbw(self, freq, vout):
        gain = np.abs(vout)
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        if valid:
            return ugbw, valid
        else:
            return freq[0], valid

    @classmethod
    def find_phm(self, freq, vout):
        gain = np.abs(vout)
        phase = np.angle(vout, deg=False)
        phase = np.unwrap(phase) # unwrap the discontinuity
        phase = np.rad2deg(phase) # convert to degrees
        #
        #plt.subplot(211)
        #plt.plot(np.log10(freq[:200]), 20*np.log10(gain[:200]))
        #plt.subplot(212)
        #plt.plot(np.log10(freq[:200]), phase)
        #IPython.embed()
        #plt.show()

        phase_fun = interp.interp1d(freq, phase, kind='quadratic')
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        if valid:
            if phase_fun(ugbw) > 0:
                return -180+phase_fun(ugbw)
            else:
                return 180 + phase_fun(ugbw)
        else:
            return -180

    @classmethod
    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop), True
        except ValueError:
            # avoid no solution
            # if abs(fzero(xstart)) < abs(fzero(xstop)):
            #     return xstart
            return xstop, False

if __name__ == '__main__':

    yname = 'bag_deep_ckt/eval_engines/spectre/specs_test/two_stage_opamp.yaml'
    eval_core = OpampMeasMan(yname)

    designs = eval_core.generate_data_set(n=1)
