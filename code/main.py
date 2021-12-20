import configargparse
import os.path as p
import numpy as np
import trimesh
from trimesh_wrapper import Trimesh_wrapper
from utils import *


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
    output = parser.add_mutually_exclusive_group(required=False)
    output.add_argument('--output-name', '-on', type=str,
                         help='module name to use as output in outputs folder')
    output.add_argument('--output-path', '-op', type=str,
                         help='module path to use as output')

    parsed = parser.parse_args()
    parsed.constraints = [1 if i in parsed.constraints else 0 for i in range(6)]
    return parsed

# ------------------------ MAIN ------------------------ #

def main() -> None:
    parser = get_parser()
    object_name = 'bike-wheel'
    mesh = trimesh.load(input_ply_path(object_name))
    mesh.apply_scale(1/max(list(mesh.bounding_box.primitive.extents)))
    # p0 = mesh.triangles_center[0]
    
    
    # for facet in mesh.facets:
    #     mesh.visual.face_colors[facet] = trimesh.visual.random_color()
    # mesh.show()
    mesh.export(output_obj_path(object_name))


if __name__ == "__main__":
    main()
    # main('bike-wheel')
    # main('copper-key')