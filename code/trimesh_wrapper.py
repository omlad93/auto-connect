from typing import List
import numpy as np
import trimesh
from trimesh.base import Trimesh
from arithmetics import *
from typing import Optional
import numpy as np
from scipy import optimize
from scipy.sparse.construct import random
from scipy.spatial.distance import hamming
from sklearn.cluster import SpectralClustering
from arithmetics import *




class Trimesh_wrapper:

    
    mesh:                   Trimesh     # original mesh (input)
    holdability_whole_mesh: float       # original mesh (input)
    convex_hull:            bool        # is original mesh a convex_hull ?
    alpha:                  float       # alpha parameter for free motions
    symmetry_planes:        list        # symmetry planes of th mesh
    symmetry_sections:      list        # symmetry sections of the mesh
    minimal_distances:      list        # minimal distances of triangles center from any symmetry plane
    current_holder:         Trimesh     # T in paper
    maximum_holder:         Trimesh     # Holder hold all legal triangles
    results:                list        # list of holders
    constraints:            list[dict]  # constraints from user
    intrinsic_free_motions: list        # computed internal free motion        
    external_free_motions:  list[list]  # free motions added by user
    triangles_to_ignore:    list[int]   # list of triangles indexes to ignore (blocking)


    def shell_init(self, starting_point:np.array) -> list:
        n = len(self.mesh.triangles_center)
        w1,w2,w3 = np.random.random_sample((3,))
        global_vec = np.random.randint(0,2,3)
        ordered_triangles = []
        sum_of_weights = weighted_sum(w1,w2,w3,starting_point,self,global_vec)
        shell_triangles = []
        for t in range(n):
            if t in self.triangles_to_ignore:
                w = np.inf
            else:
                w = 0 
                shell_triangles.append([t])
            ordered_triangles.append([t, w + sum_of_weights[t]])
        ordered_triangles.sort(key=lambda item: item[1])
        self.maximum_holder = self.mesh.submesh(shell_triangles, append = True)
        # optimization_args = (self.maximum_holder.triangles_center, self.maximum_holder.face_normals)
        # max_holder_holdability = optimize.minimize( fun=subregion_blockage, x0=[0 for i in range(6)], method='COBYLA', args=optimization_args, constraints=self.constraints)
        return (ordered_triangles)

        
    def calc_symmetry_planes(self) -> None :
        sp = []
        if self.mesh.symmetry:
            s0 = self.mesh.symmetry_section[0]
            s1 = self.mesh.symmetry_section[1]
            # get cross section of the current mesh and a planes defined by s0 s1.
            sections = [self.mesh.section(s1,s0), self.mesh.section(s0,s1)]
            for s in sections:
                p0 = extract_point(s.vertices,0)
                p4 = extract_point(s.vertices,4)
                p7 = extract_point(s.vertices,7)
                
                v = p7 - p0
                u = p4 - p0
                a,b,c = n = np.cross(u,v)
                d = np.dot(n,p7)
                sp.append([a,b,c,d])
        self.symmetry_sections = sections if self.mesh.symmetry else None
        self.symmetry_planes = sp

    def pre_process_mesh(self, constraints) -> None:
        self.calc_symmetry_planes()
        self.minimal_distances = list[d_symm(self.mesh.triangles_center, self.symmetry_planes)]
        self.constraints = []       
        self.current_holder = Trimesh       
        self.maximum_holder = Trimesh
        intrinsic_free_motions(self)                # updates self.constraints & self.intrinsic_free_motions
        external_free_motions(self,constraints)     # updates self.constraints & self.external_free_motions
        self.triangles_to_ignore = find_triangles_to_ignore(self)
    
    def export(self, path:str)->None:
        self.mesh.export(path)

    def __init__( self, mesh: Trimesh,
                  constraints: list,
                  convex_hull: bool=False,
                  alpha: int=0.5) -> None:
        self.mesh = mesh.convex_hull if convex_hull else mesh
        self.convex_hull = convex_hull
        self.alpha = alpha
        self.holdability_whole_mesh = 0
        self.pre_process_mesh(constraints)
 
## 4.1 - Shell Computation ##

def calc_starting_point(mesh_w: Trimesh_wrapper) -> np.array:
    if mesh_w.symmetry_planes:
        plane = np.random.randint(0,len(mesh_w.symmetry_sections)-1)
        point = np.random.randint(0,len(mesh_w.symmetry_sections[plane].vertices)-1)
        starting_point = mesh_w.symmetry_sections[plane].vertices[point]
        pass
    else:
        starting_point = mesh_w.mesh.triangles_center[np.random.randint(0, len( mesh_w.mesh.triangles_center)-1)]
    return starting_point
  
def shell_computation(mesh_w:Trimesh_wrapper,
                      starting_point:np.array,
                      holdability_th:float=0.1,
                      weight_th:float=np.inf) ->  Optional[list] :

    ordered_triangles = mesh_w.shell_init(starting_point)
    print_it = True
    n = len(ordered_triangles)
    
    whole_mesh_args = (mesh_w.mesh.triangles_center, mesh_w.mesh.face_normals)
    maximum_holdability = optimize.minimize( fun=subregion_blockage, x0=[0 for i in range(6)], method='COBYLA',  args=whole_mesh_args, constraints=mesh_w.constraints)
    mesh_w.holdability_whole_mesh = maximum_holdability.fun

    shell_vectors = [0 for i in range(n)]
    shell_triangles = []
    for i in range(n):
        if ordered_triangles[i][1] >= weight_th:
            print(f'\t\t > Reached infinite weight using {i} triangles: NH={normalized_holdability_value:.4f} [didn`t reach {holdability_th}]')
            break

        t = ordered_triangles[i][0]
        shell_triangles.append([t])
        shell_vectors[t] = 1
        mesh_w.current_holder = mesh_w.mesh.submesh(shell_triangles,append = True)

        optimization_args = (mesh_w.current_holder.triangles_center, mesh_w.current_holder.face_normals)
        current_holdability = optimize.minimize( fun=subregion_blockage, x0=[0 for i in range(6)], method='COBYLA', args=optimization_args, constraints=mesh_w.constraints)
        normalized_holdability_value = current_holdability.fun/mesh_w.holdability_whole_mesh
        
        if normalized_holdability_value > epsilon and print_it:
            print_it = False
            print(f'\t\t > Reached non-zero holdability for shell using {i} triangles')

        if normalized_holdability_value >= holdability_th:
            print(f'\t\t > Reached holdability threshold using {i} triangles: NH={normalized_holdability_value:.4f}')
            break

    return shell_vectors

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

def weighted_sum(w1:int, w2:int, w3:int, starting_point:np.array, mesh_w:Trimesh_wrapper, global_vec:np.array)->np.array:
    w_geod = w1 * d_geod(mesh_w.mesh.triangles_center, starting_point)
    w_symm = w2 * d_symm(mesh_w.mesh.triangles_center, mesh_w.symmetry_planes)
    w_norm = w3 * d_norm(mesh_w.mesh.face_normals, global_vec)
    return w_geod + w_symm + w_norm
    

## 4.2 - Holdability Criterion ##

def contact_blockage( point:np.array,
                      normal:np.array,
                      phi:np.ndarray,
                      b_th:float = 0
                      ) -> float:
                      
    ''' 
    calculate b(p, phi): measures the amount of blockage experienced
    by the point when the object is moved infinitesimally by φ 
    b(p, φ) = max((n^T * Γ * φ), 0)
    > Γ = ([x]^T I) is the 3x6 matrix that transforms a spatial velocity
    '''
    assert type(phi) in [np.ndarray,list], f'Phi is {type(phi)}: Phi={phi}'
    identity_matrix = np.array([
        [ 1,  0,  0 ],
        [ 0,  1,  0 ], 
        [ 0,  0,  1 ]
        ])
    x,y,z = point
    point_matrix = np.array([
        [ 0, -z,  y ],
        [ z,  0, -x ],
        [-y,  x,  0 ]
     ] )
    matrix = np.concatenate((np.transpose(point_matrix), identity_matrix), axis=1)
    try:
        b = float(np.dot(normal, np.dot(matrix, np.transpose(phi)))) 
    except:
            print (f'Normal:\n{normal}\n')
            print (f'Matrix:\n{matrix}\n')
            print (f'Phi   :\n{phi} \n')
            print (f'Shpaes: N:{normal.shape}, Matrix:{matrix.shape}, Phi:{phi.shape}' )
            exit(9)
    return b if b > b_th else 0

def subregion_blockage(phi: np.array,
                       centers:list[np.array],
                       normals:list[np.array],
                       )->float:
    ''' calculate B(T, phi) '''
    B = 0
    for i in range(len(centers)):
        B += contact_blockage(point=centers[i], normal=normals[i], phi=phi)
    return B


#unused - implemented in shell_computation for better debug 
def normalized_holdability(mesh_w:Trimesh_wrapper)->float:
    
    optimization_args = (mesh_w.current_holder.triangles_center, mesh_w.current_holder.face_normals)
    holdability = optimize.minimize(
        fun=subregion_blockage,
        x0=[0 for i in range(6)],
        method='COBYLA',
        args=optimization_args,
        constraints=mesh_w.constraints
    )     

    if not mesh_w.holdability_whole_mesh:
        whole_mesh_args = (mesh_w.mesh.triangles_center, mesh_w.mesh.face_normals)
        maximum_holdability = optimize.minimize(
            fun=subregion_blockage,
            x0=[0 for i in range(6)],
            method='COBYLA',
            args=whole_mesh_args,
            constraints=mesh_w.constraints
        )
        mesh_w.holdability_whole_mesh = maximum_holdability.fun
        denominator = maximum_holdability.fun
    else:
        denominator = mesh_w.holdability_whole_mesh
    
    assert( denominator >= holdability.fun)
    # print(f'\tcurrent value: {holdability.fun} , maximum: {denominator}')
    return (holdability.fun / denominator)


## 4.3 - Free Motions ##

def function_for_constaint(phi,x, cone_factor) -> float:
        return (rad_to_deg(angle(x,phi)) - cone_factor)

# Does Return Value Is Used ?
def intrinsic_free_motions(mesh_w: Trimesh_wrapper, cone_factor:int=30)->float:
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
    
def external_free_motions(mesh_w:Trimesh_wrapper, user_free_motions:list, cone_factor:int=30) ->None:
    user_constraints = []
    for phi_free in user_free_motions:
        assert type(phi_free) in [np.array,np.ndarray], f'type of external phi: {type(phi_free)}'
        user_constraints.append({
            'type':'ineq',
            'fun':function_for_constaint,
            'args':(phi_free, cone_factor)
        })
    mesh_w.constraints += user_constraints
    mesh_w.external_free_motions = [user_free_motions]

def find_triangles_to_ignore(mesh_w:Trimesh_wrapper) -> list[int]:
    triangles_to_ignore = []
    centers = mesh_w.mesh.triangles_center
    normals = mesh_w.mesh.face_normals
    for triangle_index in range(len(centers)):
        for phi in mesh_w.external_free_motions:
            if contact_blockage(centers[triangle_index], normals[triangle_index],phi):
                triangles_to_ignore.append(triangle_index)
    return triangles_to_ignore

## 4.5 - Providing Diverse Subset

def clustering_fit(vectors:list, results:int):
    n = len(vectors)
    similarity = np.zeros((n,n))
    for i in range (n):
        assert vectors[i] != None , f'vectors[{i}] = None'
        for j in range(i,n):
            assert vectors[j] != None , f'vectors[{j}] = None'
            assert len(vectors[i])== len(vectors[j]) , f'unmatched len: i={len(vectors[i])}   j={len(vectors[i])}'
            similarity[i,j] = similarity[j,i] = (1 - hamming(vectors[i], vectors[j]))
    clustering =  SpectralClustering(
        n_clusters=results,
        assign_labels='kmeans',
        random_state=0,
        affinity='precomputed'
        )
    return clustering.fit(similarity)

 
def extract_point (v, i:int) ->np.array:
    return (np.array((v[i][0], v[i][1], v[i][2])))
