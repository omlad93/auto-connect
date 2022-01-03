import configargparse
import os.path as p
import numpy as np
import trimesh
from trimesh_wrapper import *
from pprint import PrettyPrinter
# from utils import *

pp = PrettyPrinter(indent=4)

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
    input.add_argument('--input','-in', type=str,
                         help='module name to use as input from inputs folder')
    parser.add_argument('--iterations', '-i', type=int, default=1,
                             help='number of iterations to do')
    parser.add_argument('--results', '-r', type=int, default=1,
                             help='number of results to provide')
    parser.add_argument('--constraints', '-c', nargs='+', type=int, required=True, default=0,
                         help='a list of constraints chosen from 0 to 5')  

    parsed = parser.parse_args()
    assert parsed.results <= parsed.iterations, 'requested more results them iterations'
    parsed.constraints = np.array([[1.0 if i in parsed.constraints else 0.0 for i in range(6)]])

    print('\nArguments:\n'+
            f'\t input:       {parsed.input}\n'
            f'\t iteration:   {parsed.iterations}\n'
            f'\t results:     {parsed.results}\n'
            f'\t constraints: {parsed.constraints}\n\n'            
        )

    return parsed

# ------------------------ MAIN ------------------------ #

#TODO
def main() -> None:
    args = get_parser()
    mesh_w = Trimesh_wrapper(
        trimesh.load(input_ply_path(args.input)),
        args.constraints
    )
    results =[]
    # pp.pprint(mesh_w.__dict__)
    for iteration in range(args.iterations):
        seed = calc_starting_point(mesh_w)
        # # # # ordered_triangles = mesh_w.shell_init(seed)
        current_shell = shell_computation(mesh_w, seed) # shell_vectors list (mesh_w.current_holder is updated)
        results.append(mesh_w.current_holder)
        mesh_w.current_holder.export(output_obj_path(f'{args.input}_{iteration}'))
        print(f'type of current holder is: {type(mesh_w.current_holder)}')
        print(f' ~ Exported {output_obj_path(f"{args.input}_{iteration}")}')
        break




    mesh_w.export(output_obj_path(args.input))


if __name__ == "__main__":
    main()
    # main('bike-wheel')
    # main('copper-key')