
import numpy as np
from numpy.core.numeric import identity
from trimesh_wrapper import Trimesh_wrapper
from arithmetics import *

# function library for generating holder

def ordered_polygons(mesh: Trimesh_wrapper, free_motions: list) -> list:
    pass

## 4.1 - Shell Computation ##

def shell_computation():
    pass

def d_good():
    '''
    The Geodesic distance between barycenters of triangles t, s
    '''
    pass

def d_symm():
    '''
    The distance between triangle t and symmetry plane Pk
    use distance_from_plane(point, plane)
    '''
    pass


def d_norm():
    '''
    The angle between vector normal of triangle 't' and global direction N
    covered with 'angle' lambda expression
    '''
    pass

## 4.2 - Holdability Criterion ##


def contact_blockage( point:np.ndarray,
                      normal:np.ndarray,
                      phi:np.ndarray,
                      b_th:float
                      ):
                      
    ''' calculate b(p, phi) '''
    identity_matrix = np.asarray([
        [ 1,  0,  0 ],
        [ 0,  1,  0 ], 
        [ 0,  0,  1 ]
        ])
    x,y,z = point
    point_matrix = np.asarray([
        [ 0, -z,  y ],
        [ y,  0, -x ],
        [-y,  x,  0 ]
     ] )
    matrix = np.concatenate((np.transpose(point_matrix), identity_matrix), axis=1)
    b_vec = np.dot(np.dot(matrix, np.transpose(phi)) , normal )
    return b_vec if b_vec > b_th else 0

def blockage():
    ''' calculate B(T, phi) '''
    pass

def holdability(T):
    pass

def normalized_holdability(T):
    pass


## 4.3 - Free Motions ##


## TODO : all functions

