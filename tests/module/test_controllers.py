import math
import pypose as pp
import torch as torch

from pypose.module.dubincar_controller import DubinCarController
from pypose.module.geometric_controller import GeometricController

def test_dubincar_controller():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system_state = torch.zeros(5, device=device)
    system_ref_state = torch.ones(5, device=device)
    dubincar_controller = DubinCarController(torch.ones(4, device=device))
    controller_input = dubincar_controller.forward()


def test_geometric_controller():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    test_dubincar_controller()
    test_geometric_controller()
