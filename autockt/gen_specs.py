import numpy as np
import random
import yaml
import os
import IPython
import argparse
from collections import OrderedDict
import pickle


#way of ordering the way a yaml file is read
class OrderedDictYAMLLoader(yaml.Loader):
  """
  A YAML loader that loads mappings into ordered dictionaries.
  """

  def __init__(self, *args, **kwargs):
      yaml.Loader.__init__(self, *args, **kwargs)

      self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
      self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

  def construct_yaml_map(self, node):
      data = OrderedDict()
      yield data
      value = self.construct_mapping(node)
      data.update(value)

  def construct_mapping(self, node, deep=False):
      if isinstance(node, yaml.MappingNode):
          self.flatten_mapping(node)
      else:
          raise yaml.constructor.ConstructorError(None, None,
                                                  'expected a mapping node, but found %s' % node.id, node.start_mark)

      mapping = OrderedDict()
      for key_node, value_node in node.value:
          key = self.construct_object(key_node, deep=deep)
          value = self.construct_object(value_node, deep=deep)
          mapping[key] = value
      return mapping

#Generate the design specifications and then save to a pickle file
def gen_data(CIR_YAML, env, num_specs):
  with open(CIR_YAML, 'r') as f:
    yaml_data = yaml.load(f, OrderedDictYAMLLoader)

  specs_range = yaml_data['target_specs']
  specs_range_vals = list(specs_range.values())
  specs_valid = []
  for spec in specs_range_vals:
      if isinstance(spec[0],int):
          list_val = [random.randint(int(spec[0]),int(spec[1])) for x in range(0,num_specs)]
      else:
          list_val = [random.uniform(float(spec[0]),float(spec[1])) for x in range(0,num_specs)]
      specs_valid.append(tuple(list_val))
  i=0
  for key,value in specs_range.items():
      specs_range[key] = specs_valid[i]
      i+=1
  with open("autockt/gen_specs/ngspice_specs_gen_"+env, 'wb') as f:
    pickle.dump(specs_range,f)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_specs', type=str)
  args = parser.parse_args()
  CIR_YAML = "eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml"
  
  gen_data(CIR_YAML, "two_stage_opamp", int(args.num_specs))

if __name__=="__main__":
  main()
