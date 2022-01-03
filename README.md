# Auto-Connect
Partial implementation of auto-connect algorithm for 3D manufacturing  
Focused on free-form object holder creation (without a set of connectors)  
 * Analyze the input mesh to find intrinsic-free motions (§4.3)
 * Generate a shell that can hold the target object (§4.1 & §4.2)
 * Select a diverse subset of designs to show to the user (§4.5)  
  
Based on the following papers:  
https://koyama.xyz/project/AutoConnect/autoconnect.pdf &nbsp;&nbsp;&nbsp;: original paper  
https://koyama.xyz/project/AutoConnect/supplemental.pdf &nbsp;: supplemental material 



## Pre - Running:
1. make sure you have git LFS (https://git-lfs.github.com/)
2. Clone Auto-Connect Repo
3. install requirements: `pip install -r requirements.txt`
4. Place your `ply` file in inputs sub-folder [`inputs/module_name.ply`]



## Running Auto connect:
run `code/main.py` using terminal followed by arguments:  
* <b>`--input, -in`</b> &nbsp; <span style="color:red"> required </span>  
   module name to use as input from inputs folder 
* <b>`--iterations, -i`</b>  
   number of iterations to do (default: 1)
* <b>`--results, -r`</b>  
    number of results to provide (default: 1)
* <b>`--constraints, -c`</b> &nbsp; <span style="color:red"> required </span>   
    a list of constraints chosen from 0 to 5:  
    [0-2] : Rotational  
    [3-5] : XYZ (default: None)
* <b>`--convex_hull, -cv`</b>  
    a flag to use input`s convex hull (default: False)

## Example command lines:
* `python .\code\main.py -in teapot -c 3`
* `python .\code\main.py -c 4 -in ball -i 5 -r 1`
* `python .\code\main.py -c 4 -in copper-key -i 2 -r 1 -cv`
