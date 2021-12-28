
import numpy as np
from scipy import optimize
from numpy.core.numeric import identity
from trimesh_wrapper import Trimesh_wrapper
from arithmetics import *

# function library for generating holder

def ordered_polygons(mesh: Trimesh_wrapper, free_motions: list) -> list:
    pass

## 4.1 - Shell Computation ##

def shell_computation():
    pass

def d_geod(centers:list[np.array], seed_center:np.array) -> np.array:
    '''
    The Geodesic distance between barycenters of triangles t, s
    points_dist(centers[i], seed) - create a list
    '''
    return np.array([points_dist(centers[i],seed_center) for i in range(len(centers))])
    

def d_symm(centers:list[np.array], symmetry_planes:list) -> np.array:
    '''
    The distance between triangle t and symmetry plane Pk
    use distance_from_plane(point, plane)
    minimal distances between triangle t and  any symmetry plane Pk
    '''
    n = len(centers)
    if symmetry_planes:
        minimal_distances = []
        m = len(symmetry_planes)
        for t in range(n):
            min_distance = np.inf
            for s in range(m):
                triangle_center = centers[t]
                symmetry_plane  =  symmetry_planes[s]
                min_distance = min(min_distance, distance_from_plane(triangle_center,symmetry_plane) )
            # val = use linear computation from different mod
            minimal_distances.append(min_distance)
    else:
        minimal_distances = [0 for i in range(n)]
    return np.array(minimal_distances)

    
def d_norm(normals:list[np.array], N:np.array) -> np.array:
    '''
    The angle between vector normal of triangle 't' and global direction N
    covered with 'angle' lambda expression but should create a list 
    '''
    return np.array([angle(N,normals[i]) for i in range(len(normals))])


def weighted_sum(w1:int, w2:int, w3:int, mesh_w:Trimesh_wrapper):
    w_geod = w1 * d_geod(mesh_w.mesh.triangles_center, mesh_w.starting_point)
    w_symm = w2 * d_symm(mesh_w.mesh.triangles_center, mesh_w.symmetry_planes)
    w_norm = w3 * d_norm(mesh_w.mesh.face_normals, mesh_w.global_vec)
    return w_geod + w_symm + w_norm
    

## 4.2 - Holdability Criterion ##

def contact_blockage( point:np.ndarray,
                      normal:np.ndarray,
                      phi:np.ndarray,
                      b_th:float = 0
                      ) -> float:
                      
    ''' 
    calculate b(p, phi): T
    his value measures the amount of blockage experienced
    by the point when the object is moved infinitesimally by φ 
    '''
    identity_matrix = np.asarray([
        [ 1,  0,  0 ],
        [ 0,  1,  0 ], 
        [ 0,  0,  1 ]
        ])
    x,y,z = point
    point_matrix = np.asarray([
        [ 0, -z,  y ],
        [ z,  0, -x ],
        [-y,  x,  0 ]
     ] )
    matrix = np.concatenate((np.transpose(point_matrix), identity_matrix), axis=1)
    b = float(np.dot(normal, np.dot(matrix, np.transpose(phi))))
    return b if b > b_th else 0

def subregion_blockage(phi: np.array,
                       centers:list[np.array],
                       normals:list[np.array]
                       ):
    ''' calculate B(T, phi) '''
    B = 0
    for i in range(len(centers)):
        B += contact_blockage(centers[i], normals[i],phi)
    return B


def normalized_holdability(mesh_w:Trimesh_wrapper, constraints:list):
    optimization_args = (mesh_w.current_holder.triangles_center, mesh_w.current_holder.face_normals)
    holdability = optimize.minimize(
        fun=subregion_blockage,
        x0=[0 for i in range(6)],
        method='COBYLA',
        args=optimization_args,
        constraints=constraints
    )
    whole_mesh_args = ((mesh_w.mesh.triangles_center, mesh_w.mesh.face_normals))
    maximum_holdability = optimize.minimize(
        fun=subregion_blockage,
        x0=[0 for i in range(6)],
        method='COBYLA',
        args=whole_mesh_args,
        constraints=constraints
    )
    assert( maximum_holdability.fun >= holdability.fun)
    return (holdability.fun / maximum_holdability.fun)


## 4.3 - Free Motions ##


## TODO : all functions

