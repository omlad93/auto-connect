import numpy as np
import trimesh
from arithmetics import *
# from holder import x,y,z


def extract_point (v, i:int) ->np.array:
    return (np.array((v[i][0], v[i][1], v[i][2])))
class Trimesh_wrapper:

    mesh:               trimesh
    convex_hull:        bool
    symmetry_planes:    list
    minimal_distances:  list
    starting_point:     int # index in surfaces list of starting point


    def calc_minimal_distances(self) -> None:
        self.minimal_distances = []
        n = len(self.mesh.triangles_center)
        m = len(self.symmetry_planes)
        for t in range(n):
            min_distance = np.inf
            for s in range(m):
                triangle_center = self.mesh.triangles_center[t]
                symmetry_plane  =  self.symmetry_planes[s]
                min_distance = min(min_distance, distance_from_plane(triangle_center,symmetry_plane) )
            # val = use linear computation from different mod
            self.minimal_distances.append(min_distance)
        print(self.minimal_distances)

    def calc_starting_point(self) -> None:
        lst = self.calc_symmetry_planes()
        starting_point_idx = 0
        if lst:
            pass
        else:
            pass
            # return get_rand_point()
        self.starting_point = starting_point_idx
        
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

        self.symmetry_planes = sp

    def pre_process_mesh(self) -> None:
        self.calc_symmetry_planes()
        pass

    def export(self, path:str)->None:
        self.mesh.export(path)

    def __init__( self, mesh:trimesh,
                  convex_hull: bool=False) -> None:
        self.mesh = mesh
        self.convex_hull = convex_hull
        self.pre_process_mesh()

        # self.calc_symmetry_planes()
        # self.calc_starting_point()