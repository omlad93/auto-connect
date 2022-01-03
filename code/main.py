import configargparse
import os
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
                         help='a list of constraints chosen from 0 to 5: 3-X, 4-Y, 5-Z')  

    parsed = parser.parse_args()
    assert parsed.results <= parsed.iterations, 'requested more results them iterations'
    assert  parsed.results >= 0 and  parsed.iterations >=0 , 'need a none zero iteration, result count'
    parsed.constraints = np.array([[1.0 if i in parsed.constraints else 0.0 for i in range(6)]])

    print(  f'\n Auto-Connect run for {args.input}:\n'+
            f'\t iterations:   {parsed.iterations}\n'
            f'\t results:     {parsed.results}\n'
            f'\t constraints: {parsed.constraints}\n'            
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
    vectors =[]
    # pp.pprint(mesh_w.__dict__)
    os.makedirs(f'outputs{p.sep}{args.input}{p.sep}all', exist_ok=True)
    os.makedirs(f'outputs{p.sep}{args.input}{p.sep}clustered', exist_ok=True)
    for iteration in range(args.iterations):
        print(f'computing shell #{iteration}:')
        seed = calc_starting_point(mesh_w)
        current_shell = shell_computation(mesh_w, seed)
        results.append(mesh_w.current_holder)
        vectors.append(current_shell)
        out_path = output_obj_path(f'{args.input}{p.sep}all{p.sep}{args.input}_{iteration}')
        mesh_w.current_holder.export(out_path)
        print(f'\t ~ Export: {out_path}')
    if args.iterations > args.results:
        clustered = [False for i in range(args.results)]
        print(f'\nrunning Clustering {args.iterations}->{args.results}:')
        clustering = clustering_fit(vectors, args.results)
        for i in range (len(vectors)):
            for j in range (i,len(vectors)):
                if clustering.labels_[j] == i:
                    if not clustered[i]:
                        out_path = output_obj_path(f'{args.input}{p.sep}clustered{p.sep}{args.input}_{i}')
                        results[i].export(out_path)
                        print(f'\t ~ Export: {out_path}')
                        clustered[i] = True

        print(f'\nAuto-Connect run for {args.input} has finished.\n')







if __name__ == "__main__":
    main()
    # main('bike-wheel')
    # main('copper-key')