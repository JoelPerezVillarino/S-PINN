import numpy as np

def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred, ord=2) / np.linalg.norm(y_true, ord=2)

def l1_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred, ord=1) / np.linalg.norm(y_true, ord=1)

def linf_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred, ord=np.inf) / np.linalg.norm(y_true, ord=np.inf)

def real_l1_relative_error(y_true, y_pred):
    term = np.abs(y_true - y_pred) / np.abs(y_true)
    return np.linalg.norm(term, ord=1)

def real_l2_relative_error(y_true, y_pred):
    term = np.abs(y_true - y_pred) / np.abs(y_true)
    return np.linalg.norm(term, ord=2)

def real_linf_relative_error(y_true, y_pred):
    term = np.abs(y_true - y_pred) / np.abs(y_true)
    return np.linalg.norm(term, ord=np.inf)

def mean_l2_relative_error(y_true, y_pred):
    return np.mean(
        np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(y_true, axis=1)
    )

def _absolute_percentage_error(y_true, y_pred):
    return 100. * np.abs(
        (y_true - y_pred) / np.clip(np.abs(y_true), np.finfo(np.float64).eps, None)
    )

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(_absolute_percentage_error(y_true, y_pred))

def max_absolute_percentage_error(y_true, y_pred):
    return np.amax(_absolute_percentage_error(y_true, y_pred))

def absolute_percentage_error_std(y_true, y_pred):
    return np.std(_absolute_percentage_error(y_true, y_pred))



def get(identifier):
    metric_identifier = {
        "l1 relative error": l1_relative_error,
        "l2 relative error": l2_relative_error,
        "linf relative error": linf_relative_error,
        "real l1 relative error": real_l1_relative_error,
        "real l2 relative error": real_l2_relative_error,
        "real linf relative error": real_linf_relative_error,
        "mean l2 relative error": mean_l2_relative_error,
        "MAPE": mean_absolute_percentage_error,
        "max APE": max_absolute_percentage_error,
        "APE SD": absolute_percentage_error_std,
    }

    if isinstance(identifier, str):
        return metric_identifier[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret metric function identifier:", identifier)