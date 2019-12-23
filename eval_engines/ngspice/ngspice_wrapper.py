import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import abc
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import time
import pprint

debug = False

class NgSpiceWrapper(object):

    BASE_TMP_DIR = os.path.abspath("/tmp/ckt_da")

    def __init__(self, num_process, design_netlist, root_dir=None):
        if root_dir == None:
            self.root_dir = NgSpiceWrapper.BASE_TMP_DIR
        else:
            self.root_dir = root_dir
        _, dsg_netlist_fname = os.path.split(design_netlist)
        self.base_design_name = os.path.splitext(dsg_netlist_fname)[0]
        self.num_process = num_process
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)

        raw_file = open(design_netlist, 'r')
        self.tmp_lines = raw_file.readlines()
        raw_file.close()


    def get_design_name(self, state):
        fname = self.base_design_name
        for value in state.values():
            fname += "_" + str(value)
        return fname

    def create_design(self, state, new_fname):
        design_folder = os.path.join(self.gen_dir, new_fname)
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.cir')

        lines = copy.deepcopy(self.tmp_lines)
        for line_num, line in enumerate(lines):
            if '.include' in line:
                regex = re.compile("\.include\s*\"(.*?)\"")
                found = regex.search(line)
                if found:
                    # current_fpath = os.path.realpath(__file__)
                    # parent_path = os.path.abspath(os.path.join(current_fpath, os.pardir))
                    # parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
                    # path_to_model = os.path.join(parent_path, 'spice_models/45nm_bulk.txt')
                    # lines[line_num] = lines[line_num].replace(found.group(1), path_to_model)
                    pass # do not change the model path
            if '.param' in line:
                for key, value in state.items():
                    regex = re.compile("%s=(\S+)" % (key))
                    found = regex.search(line)
                    if found:
                        new_replacement = "%s=%s" % (key, str(value))
                        lines[line_num] = lines[line_num].replace(found.group(0), new_replacement)
            if 'wrdata' in line:
                regex = re.compile("wrdata\s*(\w+\.\w+)\s*")
                found = regex.search(line)
                if found:
                    replacement = os.path.join(design_folder, found.group(1))
                    lines[line_num] = lines[line_num].replace(found.group(1), replacement)

        with open(fpath, 'w') as f:
            f.writelines(lines)
            f.close()
        return design_folder, fpath

    def simulate(self, fpath):
        info = 0 # this means no error occurred
        command = "ngspice -b %s >/dev/null 2>&1" %fpath
        exit_code = os.system(command)
        if debug:
            print(command)
            print(fpath)

        if (exit_code % 256):
           # raise RuntimeError('program {} failed!'.format(command))
            info = 1 # this means an error has occurred
        return info


    def create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        if debug:
            print('state', state)
            print('verbose', verbose)
        if dsn_name == None:
            dsn_name = self.get_design_name(state)
        else:
            dsn_name = str(dsn_name)
        if verbose:
            print(dsn_name)
        design_folder, fpath = self.create_design(state, dsn_name)
        info = self.simulate(fpath)
        specs = self.translate_result(design_folder)
        return state, specs, info


    def run(self, states, design_names=None, verbose=False):
        """

        :param states:
        :param design_names: if None default design name will be used, otherwise the given design name will be used
        :param verbose: If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value), info: int)]
        """
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name)in zip(states, design_names)]
        specs = pool.starmap(self.create_design_and_simulate, arg_list)
        pool.close()
        return specs

    def translate_result(self, output_path):
        """
        This method needs to be overwritten according to cicuit needs,
        parsing output, playing with the results to get a cost function, etc.
        The designer should look at his/her netlist and accordingly write this function.

        :param output_path:
        :return:
        """
        result = None
        return result


"""
this is an example of using NgSpiceWrapper for a cs_amp ac and dc simulation
Look at the cs_amp.cir as well to make sense out of the parser.
When you run the template netlist it will generate the data in a way that can be easily handled with the following
class methods.
"""
class CsAmpClass(NgSpiceWrapper):

    def translate_result(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # use parse output here
        freq, vout, Ibias = self.parse_output(output_path)
        bw = self.find_bw(vout, freq)
        gain = self.find_dc_gain(vout)


        spec = dict(
            bw=bw,
            gain=gain,
            Ibias=Ibias
        )

        return spec

    def parse_output(self, output_path):

        ac_fname = os.path.join(output_path, 'ac.csv')
        dc_fname = os.path.join(output_path, 'dc.csv')

        if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
            print("ac/dc file doesn't exist: %s" % output_path)

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout = ac_raw_outputs[:, 1]
        ibias = -dc_raw_outputs[1]

        return freq, vout, ibias

    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    def find_bw(self, vout, freq):
        gain = np.abs(vout)
        gain_3dB = gain[0] / np.sqrt(2)
        return self._get_best_crossing(freq, gain, gain_3dB)


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
class CsAmpEvaluationCore(object):

    def __init__(self, cir_yaml):
        import yaml
        with open(cir_yaml, 'r') as f:
            yaml_data = yaml.load(f)

        # specs
        specs = yaml_data['target_specs']
        self.bw_min     = specs['bw_min']
        self.gain_min   = specs['gain_min']
        self.bias_max   = specs['ibias_max']

        num_process = yaml_data['num_process']
        dsn_netlist = yaml_data['dsn_netlist']
        self.env = CsAmpClass(num_process=num_process, design_netlist=dsn_netlist)

        params = yaml_data['params']
        self.res_vec = np.arange(params['rload'][0], params['rload'][1], params['rload'][2])
        self.mul_vec = np.arange(params['mul'][0], params['mul'][1], params['mul'][2])

    def cost_fun(self, res_idx, mul_idx, verbose=False):
        """

        :param res:
        :param mul:
        :param verbose: if True will print the specification performance of the best individual
        :return:
        """
        state = [{'rload': self.res_vec[res_idx], 'mul': self.mul_vec[mul_idx]}]
        results = self.env.run(state, verbose=False)
        bw_cur = results[0][1]['bw']
        gain_cur = results[0][1]['gain']
        ibias_cur = results[0][1]['Ibias']

        if verbose:
            print('bw = %f vs. bw_min = %f' %(bw_cur, self.bw_min))
            print('gain = %f vs. gain_min = %f' %(gain_cur, self.gain_min))
            print('Ibias = %f vs. Ibias_max = %f' %(ibias_cur, self.bias_max))

        cost = 0
        if bw_cur < self.bw_min:
            cost += abs(bw_cur/self.bw_min - 1.0)
        if gain_cur < self.gain_min:
            cost += abs(gain_cur/self.gain_min - 1.0)
        cost += abs(ibias_cur/self.bias_max)/10

        return cost
## helper function for demonstration

def cost_fc(ibias_cur, gain_cur, bw_cur):
    bw_min = 1e9
    gain_min = 3
    bias_max = 1e-3

    cost = 0
    if bw_cur < bw_min:
        cost += abs(bw_cur/bw_min - 1.0)
    if gain_cur < gain_min:
        cost += abs(gain_cur/gain_min - 1.0)
    cost += abs(ibias_cur/bias_max)/10

    return cost

def load_array(fname):
    with open(fname, "rb") as f:
        arr = np.load(f)
    return arr

def save_array(fname, arr):
    with open(fname, "wb") as f:
        np.save(f, arr)
    return arr

if __name__ == '__main__':

    """
    example usage of CsAmpClass we just wrote:
    This common source is comprised of a single nmos transistor and single resistor.
    The parameters are Width, Length and multiplier of the nmos, resistor value, VGS bias of the nmos.
    Cload is assumed to be given. The goal is to satisfy some functionality specs (gain_min, bw_min) while
    getting the minimum ibias as the objective of the optimization. This section of code only illustrates
    how to generate the data and validate different points in the search space.
    
    """


    # test the CsAmp class with parameter values as input
    num_process = 1
    dsn_netlist = './framework/netlist/cs_amp.cir'
    cs_env = CsAmpClass(num_process=num_process, design_netlist=dsn_netlist)

    # example of running it for one example point and getting back the data
    state_list = [{'mul': 5, 'rload': 1630}]
    results = cs_env.run(state_list, verbose=False)
    if debug:
        print(results)

    # test Evaluation core and cost function with indices as input
    eval_core = CsAmpEvaluationCore("./framework/yaml_files/cs_amp.yaml")
    # let's say we want to evaluate some point (opt_res, opt_mul)
    opt_res = 85
    opt_mul = 9
    cost = eval_core.cost_fun(res_idx=opt_res, mul_idx=opt_mul, verbose=True)
    print(cost)

    ##  generate the two-axis grid world of the cs_amp example: (rload, mul) in each cell
    #   store bw, gain, Ibias and store them in file for later use, plot the specs:
    gen_data = False
    verbose = True

    if gen_data:
        mul_vec = np.arange(1, 100, 1)
        res_vec = np.arange(10, 5000, 20)
        result_list = []

        for i, mul in enumerate(mul_vec):
            for j, res in enumerate(res_vec):
                state_list = [{'mul': mul, 'rload': res}]
                results = cs_env.run(state_list, verbose=verbose)
                if (i==6 and j==85 and debug):
                    print(results[0][1])
                result_list.append(results[0][1])

        Ibias_vec = [result['Ibias'] for result in result_list]
        bw_vec = [result['bw'] for result in result_list]
        gain_vec = [result['gain'] for result in result_list]
        # print(mul_vec)
        # print(res_vec)
        # print (len(Ibias_vec))
        # print (gain_vec)

        Ibias_mat = np.reshape(Ibias_vec, [len(mul_vec), len(res_vec)])
        bw_mat = np.reshape(bw_vec, [len(mul_vec), len(res_vec)])
        gain_mat = np.reshape(gain_vec, [len(mul_vec), len(res_vec)])

        # the data is going to be mul as x and res as y
        save_array("./genome/sweeps/bw.array", bw_mat)
        save_array("./genome/sweeps/gain.array", gain_mat)
        save_array("./genome/sweeps/ibias.array", Ibias_mat)
        save_array("./genome/sweeps/mul_vec.array", mul_vec)
        save_array("./genome/sweeps/res_vec.array", res_vec)


    if not gen_data:
        bw_mat =    load_array("./genome/sweeps/bw.array")
        gain_mat =  load_array("./genome/sweeps/gain.array")
        Ibias_mat = load_array("./genome/sweeps/ibias.array")
        mul_vec =   load_array("./genome/sweeps/mul_vec.array")
        res_vec =   load_array("./genome/sweeps/res_vec.array")

    # Plotting the data
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm

    mul_mat, res_mat = np.meshgrid(mul_vec, res_vec, indexing='ij')

    bw_fun = interp.interp2d(res_vec,   mul_vec, bw_mat, kind="cubic")
    bias_fun = interp.interp2d(res_vec, mul_vec, Ibias_mat, kind="cubic")
    gain_fun = interp.interp2d(res_vec, mul_vec, gain_mat, kind="cubic")


    Ib_min = np.min(np.min(Ibias_mat))
    Ib_max = np.max(np.max(Ibias_mat))
    gain_min = np.min(np.min(gain_mat))
    gain_max = np.max(np.max(gain_mat))
    bw_min = np.min(np.min(bw_mat))
    bw_max = np.max(np.max(bw_mat))


    fig = plt.figure()
    fig.suptitle("(opt_res,opt_mul)=(%d,%d)" %(opt_res, opt_mul))

    ax = fig.add_subplot(221)
    mappable = ax.pcolormesh(np.log10(Ibias_mat*1e3), cmap='OrRd', vmin=np.log10(Ib_min*1e3), vmax=np.log10(Ib_max*1e3))
    ax.plot(opt_res, opt_mul, 'x', color='k')
    text = "%.2fmA" %(Ibias_mat[opt_mul][opt_res]*1e3)
    ax.annotate(text, xy=(opt_res, opt_mul),
                xytext=(opt_res+10, opt_mul+2))
    ax.axis([0, len(res_vec)-1, 0, len(mul_vec)-1])
    plt.colorbar(mappable)
    ax.set_xlabel('res_idx')
    ax.set_ylabel('mul_idx')
    ax.set_title('log10(Ibias*1e3)')

    ax = fig.add_subplot(222)
    mappable = ax.pcolormesh(np.log10(gain_mat), cmap='OrRd', vmin=np.log10(gain_min), vmax=np.log10(gain_max))
    ax.plot(opt_res, opt_mul, 'x', color='k')
    text = "%.2f" %(gain_mat[opt_mul][opt_res])
    ax.annotate(text, xy=(opt_res, opt_mul),
                xytext=(opt_res+10, opt_mul+2))
    ax.axis([0, len(res_vec)-1, 0, len(mul_vec)-1])
    plt.colorbar(mappable)
    ax.set_xlabel('res_idx')
    ax.set_ylabel('mul_idx')
    ax.set_title('log10(gain)')


    ax = fig.add_subplot(223)
    mappable = ax.pcolormesh(np.log10(bw_mat), cmap='OrRd', vmin=np.log10(bw_min), vmax=np.log10(bw_max))
    ax.plot(opt_res, opt_mul, 'x', color='k')
    text = "%.2fGHz" %(bw_mat[opt_mul][opt_res]/1e9)
    ax.annotate(text, xy=(opt_res, opt_mul),
                xytext=(opt_res+10, opt_mul+2))
    ax.axis([0, len(res_vec)-1, 0, len(mul_vec)-1])
    plt.colorbar(mappable)
    ax.set_xlabel('res_idx')
    ax.set_ylabel('mul_idx')
    ax.set_title('log10(bw[Hz])')


    cost_mat = [[cost_fc(Ibias_mat[mul_idx][res_idx],
                         gain_mat[mul_idx][res_idx],
                         bw_mat[mul_idx][res_idx]) for res_idx in \
                 range(len(res_vec))] for mul_idx in range(len(mul_vec))]

    cost_min = np.min(np.min(cost_mat))
    cost_max = np.max(np.max(cost_mat))

    ax = fig.add_subplot(224)
    # mappable = ax.pcolormesh(np.log10(cost_mat), cmap='OrRd', vmin=np.log10(cost_min), vmax=np.log10(cost_max))
    mappable = ax.pcolormesh(cost_mat, cmap='OrRd', vmin=cost_min, vmax=cost_max)
    ax.plot(opt_res, opt_mul, 'x', color='k')
    text = "%.2f" %(cost_mat[opt_mul][opt_res])
    ax.annotate(text, xy=(opt_res, opt_mul),
                xytext=(opt_res+10, opt_mul+2))
    ax.axis([0, len(res_vec)-1, 0, len(mul_vec)-1])
    plt.colorbar(mappable)
    ax.set_xlabel('res_idx')
    ax.set_ylabel('mul_idx')
    ax.set_title('cost')
    ax.grid()
    fig.tight_layout()









