import numpy as np
import tensorflow as tf
from pyDOE import lhs


def createUniformMesh2D(xGrid: np.ndarray, tauGrid: np.ndarray, precision: str):
    xx, tt = np.meshgrid(xGrid, tauGrid)
    points = np.vstack((np.ravel(xx), np.ravel(tt))).T
    n, shape = points.shape[0], xx.shape
    points = tf.Variable(points, trainable=False, dtype=precision)
    return points, n, shape

def createUniformMesh3D(xGrid: np.ndarray, yGrid: np.ndarray, tauGrid: np.ndarray, precision: str):
    xx, yy, tt = np.meshgrid(xGrid, yGrid, tauGrid)
    points = np.vstack((np.ravel(xx), np.ravel(yy), np.ravel(tt))).T
    n, shape = points.shape[0], xx.shape
    points = tf.Variable(points, trainable=False, dtype=precision)
    return points, n, shape

def createLatinHypercubeMesh(vars_min: np.ndarray, vars_max: np.ndarray, n_samples: int, precision: str):
    n_factors = len(vars_min)
    if n_factors != len(vars_max):
        raise ValueError("Vars_min and vars_max must be 1d-arrays with the same dimension")
    y = lhs(n_factors, n_samples)
    samples = vars_min + y * (vars_max - vars_min)
    samples = tf.Variable(samples, trainable=False, dtype=precision)
    return samples

def createSlices2D(xGrid: np.ndarray, tauGrid: np.ndarray, precision: str):
    pointsList = []
    for j, tau in enumerate(tauGrid):
        xx, tt = np.meshgrid(xGrid, np.array([tau]))
        points = np.vstack((np.ravel(xx), np.ravel(tt))).T
        #points = tf.Variable(points, trainable=False, dtype=precision)
        pointsList.append(points)
    pointsList = tf.convert_to_tensor(pointsList)
    return pointsList, xGrid.size * len(tauGrid), (xGrid.size, len(pointsList))
