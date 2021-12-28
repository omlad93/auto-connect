import configargparse
import os.path as p
import numpy as np
import trimesh
from trimesh_wrapper import Trimesh_wrapper
from utils import *


# • Analyze the input mesh to find intrinsic-free motions (§4.3).
# • Generate a shell that can hold the target object (§4.1 & §4.2).
# • Select a diverse subset of designs to show to the user (§4.5).

def file_path(filename:str, sub_dir:str, file_type:str='') -> str:
    if sub_dir in ['inputs','i']:
        return f'{p.curdir}{p.sep}{"inputs"}{p.sep}{filename}.{file_type if file_type else "ply"}'
    elif sub_dir in ['outputs','o']:
        return f'{p.curdir}{p.sep}{"outputs"}{p.sep}{filename}.{file_type if file_type else "obj"}'
    else:
        return f'{p.curdir}{p.sep}{filename}'


def input_ply_path(ply_name:str) -> str:
    return file_path(ply_name,'i')


def output_obj_path(obj_name:str) -> str:
    return file_path(obj_name,'o')


# ----------------------- PARSER ----------------------- #

def get_parser() -> configargparse.ArgumentParser:
    parser = configargparse.get_argument_parser()

    input = parser.add_mutually_exclusive_group(required=True)
    input.add_argument('--input-name','-in', type=str,
                         help='module name to use as input from inputs folder')
    input.add_argument('--input-path','-ip', type=str,
                         help='module path to use as input')

    parser.add_argument('--iterations-count', '-i', type=int, default=1,
                             help='number of iterations to do')
    parser.add_argument('--results-count', '-r', type=int, default=1,
                             help='number of results to provide')
    parser.add_argument('--constraints', '-c', nargs='+', type=int, required=True, default=0,
                         help='a list of constraints chosen from 0 to 5')  

    parsed = parser.parse_args()
    parsed.constraints = [1 if i in parsed.constraints else 0 for i in range(6)]
    parsed.input = parsed.input_name if parsed.input_name else parsed.input_path

    return parsed

# ------------------------ MAIN ------------------------ #

#TODO
def main() -> None:
    parser = get_parser()
    input = 'ball'
    mesh_w = Trimesh_wrapper(trimesh.load(input_ply_path(input)))
    mesh_w.calc_symmetry_planes()
    mesh_w.calc_minimal_distances()

    #mesh.apply_scale(1/max(list(mesh.bounding_box.primitive.extents)))
    # p0 = mesh.triangles_center[0]
    
    
    # for facet in mesh.facets:
    #     mesh.visual.face_colors[facet] = trimesh.visual.random_color()
    # mesh.show()
    mesh_w.export(output_obj_path(input))


if __name__ == "__main__":
    main()
    # main('bike-wheel')
    # main('copper-key')