import main
import numpy as np


def test_step():
    X = np.array([-1.0, 1.0, 2.0])
    assert main.step_function(X)[0] == 0
    assert main.step_function(X)[1] == 1
    assert main.step_function(X)[2] == 1
