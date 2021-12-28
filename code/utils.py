
import numpy as np
from scipy import optimize
from scipy.sparse.construct import random
from scipy.spatial.distance import hamming
from sklearn.cluster import SpectralClustering
from trimesh_wrapper import Trimesh_wrapper
from arithmetics import *

# function library for generating holder

#TODO
def ordered_polygons(mesh: Trimesh_wrapper, free_motions: list) -> list:
    pass

## 4.1 - Shell Computation ##

#TODO
def shell_computation():
    # init random variables
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

def weighted_sum(w1:int, w2:int, w3:int, mesh_w:Trimesh_wrapper)->np.array:
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
    calculate b(p, phi): measures the amount of blockage experienced
    by the point when the object is moved infinitesimally by φ 
    b(p, φ) = max((n^T * Γ * φ), 0)
    > Γ = ([x]^T I) is the 3x6 matrix that transforms a spatial velocity
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

def subregion_blockage(centers:list[np.array],
                       normals:list[np.array],
                       phi: np.array
                       )->float:
    ''' calculate B(T, phi) '''
    B = 0
    for i in range(len(centers)):
        B += contact_blockage(centers[i], normals[i], phi)
    return B

def normalized_holdability(mesh_w:Trimesh_wrapper, constraints:list)->float:
    optimization_args = (mesh_w.current_holder.triangles_center, mesh_w.current_holder.face_normals)
    holdability = optimize.minimize(
        fun=subregion_blockage,
        x0=[0 for i in range(6)],
        method='COBYLA',
        args=optimization_args,
        constraints=constraints
    )
    whole_mesh_args = (mesh_w.mesh.triangles_center, mesh_w.mesh.face_normals)
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

def function_for_constaint(x, cone_factor, phi) -> float:
        return (rad_to_deg(angle(x,phi)) - cone_factor)

def intrinsic_free_motions(mesh_w: Trimesh_wrapper, cone_factor:int)->float:
    '''
    In some cases, the holdability measure is zero even if all triangles of
    the mesh are included in the triangle set T
    '''
    def sumsq_minus_one(input):
        return (np.sum(np.square(input)) - 1)

    def one_minus_sumsq(input):
        return (1 - np.sum(np.square(input)))
    

    intrinsic_free_motions = []
    constraints = [{'type':'ineq', 'fun':one_minus_sumsq}, {'type':'ineq', 'fun':sumsq_minus_one}]
    args = (mesh_w.mesh.triangles_center, mesh_w.mesh.face_normals)
    min_val = 0
    while min_val == 0:
        optimization = optimize.minimize(
            fun=subregion_blockage,
            x0=[0 for i in range(6)],
            method='COBYLA',
            args=args,
            constraints=constraints)
        min_val = optimization.fun
        if min_val==0:
            intrinsic_free_motions.append(optimization.x)
            constraints.append(
                {
                    'type':'ineq',
                    'fun': function_for_constaint,
                    'args': (optimization.x, cone_factor)
                }
            )
    mesh_w.constraints += constraints 
    mesh_w.intrinsic_free_motions = intrinsic_free_motions
    return min_val
    
def external_free_motions(mesh_w:Trimesh_wrapper, user_free_motions:list, cone_factor:int) ->None:
    user_constraints = []
    for phi_free in user_free_motions:
        user_constraints.append({
            'type':'ineq',
            'fun':function_for_constaint,
            'args':(np.array(phi_free), cone_factor)
        })
    mesh_w.constraints += user_constraints
    mesh_w.external_free_motions = user_free_motions

def find_triangles_to_ignore(mesh_w:Trimesh_wrapper):
    triangles_to_ignore = []
    centers = mesh_w.mesh.triangles_center
    normals = mesh_w.mesh.face_normals
    for triangle_index in range(len(centers)):
        for phi in mesh_w.external_free_motions:
            if contact_blockage(centers[triangle_index], normals[triangle_index],phi):
                triangles_to_ignore.append(triangle_index)
    mesh_w = triangles_to_ignore

## 4.5 - Providing Diverse Subset

def clustering_fit(vectors:list, results:int):
    n = len(vectors)
    similarity = np.zeros((n,n))
    for i in range (n):
        for j in range(i,n):
            similarity[i,j] = similarity[j,i] = (1 - hamming(vectors[i], vectors[j]))
    clustering =  SpectralClustering(
        n_clusters=results,
        assign_labels='kmeans',
        random_state=0,
        affinity='precomputed'
        )
    return clustering.fit(similarity)
    