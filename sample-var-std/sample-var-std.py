import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    var=float((np.sum((np.mean(x)-np.array(x))**2))/(len(x)-1))
    std=float(np.sqrt(var))
    return {"variance":var, "standard_deviation": std}