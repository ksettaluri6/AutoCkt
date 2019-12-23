import sys
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt
import libpsf
import fnmatch
import pdb
import IPython

IGNORE_LIST = ['*.info', '*.primitives', '*.subckts']

class FileNotCompatible(Exception):
    """
    raise when file is not compatible with libpsf
    """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self,  args, kwargs)

def is_ignored(string):
    return any([fnmatch.fnmatch(string, pattern) for pattern in IGNORE_LIST])

class SpectreParser(object):

    @classmethod
    def parse(cls, raw_folder):
        folder_path = os.path.abspath(raw_folder)
        data = dict()
        files =  os.listdir(folder_path)
        for file in files:
            if is_ignored(file):
                continue
            try:
                file = os.path.join(raw_folder, file)
                datum = cls.process_file(file)
            except FileNotCompatible:
                # print('failed on {}'.format(file))
                continue

            _, kwrd = os.path.split(file)
            kwrd = os.path.splitext(kwrd)[0]
            data[kwrd] = datum

        return data

    @classmethod
    def process_file(cls, file):
        fpath = os.path.abspath(file)
        try:
            psfobj = libpsf.PSFDataSet(fpath)
        except:
            raise FileNotCompatible('file {} was not compatible with libpsf'.format(file))

        is_swept = psfobj.is_swept()
        datum = dict()
        for signal in psfobj.get_signal_names():
            datum[signal] = psfobj.get_signal(signal)

        if is_swept:
            datum['sweep_vars'] = psfobj.get_sweep_param_names()
            datum['sweep_values'] = psfobj.get_sweep_values()

        psfobj.close()
        return datum


if __name__ == '__main__':

   res = SpectreParser.parse(sys.argv[1])
   pdb.set_trace()
   print(res)
