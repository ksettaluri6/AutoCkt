import numpy as np
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt
import yaml
import importlib

from eval_engines.ngspice.TwoStageComplete import TwoStageMeasManager

class NoGainBoostMeasManager(TwoStageMeasManager):

    def _get_specs(self, results_dict):
        ugbw_cur = results_dict['ol'][0][1]['ugbw']
        gain_cur = results_dict['ol'][0][1]['gain']
        phm_cur = results_dict['ol'][0][1]['phm']
        ibias_cur = results_dict['ol'][0][1]['Ibias']

        specs_dict = dict(
            gain=gain_cur,
            ugbw=ugbw_cur,
            pm=phm_cur,
            ibias=ibias_cur,
        )

        return specs_dict


