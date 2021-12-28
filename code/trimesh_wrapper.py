import numpy as np
import trimesh
from arithmetics import *
from utils import *
# from holder import x,y,z


def extract_point (v, i:int) ->np.array:
    return (np.array((v[i][0], v[i][1], v[i][2])))
class Trimesh_wrapper:

    
    mesh:               trimesh     # original mesh (input)
    convex_hull:        bool        # is original mesh a convex_hull ?
    symmetry_planes:    list        
    minimal_distances:  list        # minimal distances of triangles center from any symmetry plane
    current_holder:     trimesh     # T in paper
    starting_point:     np.array    # of generating current holder
    global_vec:         np.array    # for weighted sum calculation
    results:            list        # list of holders




    #TODO
    def calc_starting_point(self) -> None:
        lst = self.calc_symmetry_planes()
        starting_point_idx = 0
        if self.symmetry_planes:
            # find point closest to symmetry plane
            pass
        else:
            # get random point
            pass
            # return ()
        
        # self.starting_point = starting_point_idx
        
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

    # NOT DONE
    def pre_process_mesh(self) -> None:
        self.calc_symmetry_planes()
        self.minimal_distances = list[d_symm(self.mesh.triangles_center, self.symmetry_planes)]
        self.global_vec = [0,0,-1] # Gravity (For Now)
        self.calc_starting_point()
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