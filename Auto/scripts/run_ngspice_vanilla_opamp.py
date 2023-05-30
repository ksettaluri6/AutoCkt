if __name__ != "__main__":
    raise Exception("This is a SCRIPT and should be run as __main__!")

from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp
import IPython


def main():
    env_config = {"generalize": True, "valid": True}
    env = TwoStageAmp(env_config)
    env.reset()
    env.step([2, 2, 2, 2, 2, 2, 2])
    IPython.embed()


main()
