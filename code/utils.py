
import numpy as np
from numpy.core.numeric import identity
from trimesh_wrapper import Trimesh_wrapper

# function library for generating holder


epsilon     = 10**-5
size        = lambda v : np.sqrt(np.sum(np.dot(v,v)))
angle       = lambda v1,v2 : np.arccos(np.dot(v1,v2)/(size(v1)*size(v2)))
zero_clip   = lambda x : max(x,epsilon)
points_diff = lambda p1,p2 : [p1[i] - p2[i] for i in range(len(p1))]
points_dist = lambda p1,p2 : np.sqrt(np.sum(np.square(points_diff(p1,p2)))) 

def distance_from_plane(point:np.ndarray, plane: np.ndarray) -> float:
    ''' 
        A function to find minimal distance of point from plane
        point = [px, py, pz]
        plane = [a, b, c, d] 
    '''
    x,y,z = point
    a,b,c,d = plane
    numerator = abs (a*x + b*y + c*z + d)
    denominator = size(np.array(a,b,c))
    return numerator/denominator if (denominator != 0) else np.inf


def ordered_polygons(mesh: Trimesh_wrapper, free_motions: list) -> list:
    pass

## 4.1 - Shell Computation ##

def shell_computation():
    pass

def d_good():
    pass

def d_symm():
    pass

def d_norm():
    pass

## 4.2 - Holdability Criterion ##


def contact_blockage(point:np.ndarray, normal:np.ndarray, phi:np.ndarray, b_th:float):
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

