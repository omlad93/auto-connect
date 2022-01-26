# Auto-Connect
Partial implementation of auto-connect algorithm for 3D manufacturing  
Focused on free-form object holder creation (without a set of connectors)  
 * Analyze the input mesh to find intrinsic-free motions (§4.3)
 * Generate a shell that can hold the target object (§4.1 & §4.2)
 * Select a diverse subset of designs to show to the user (§4.5)  

 <span style="color:green"> <b> Our implementation is Naive and meant to be descriptive (follow the steps on the paper) </b> </span>

Based on the following papers:  
https://koyama.xyz/project/AutoConnect/autoconnect.pdf &nbsp;&nbsp;&nbsp;: original paper  
https://koyama.xyz/project/AutoConnect/supplemental.pdf &nbsp;: supplemental material 

## Disclaimer :
The output meshes <b> are not printable </b> due to infinitesimal width:  
as a result of the algorithm method of choosing triangles from the shell of the original mesh. This can be mitigated in two methods:  
1. Manually editing the output in a 3D-editor (we used Blender)
2. improving the algorithm beyond what is described in the paper.  
   (we found a naive algorithm described below: had no time to implement)  

We understand that our disclaimer strongly demonstrates the take-off message of the course:  
<b>`It's in the power of programmers to make 3D manufacturing accessible`</b>  
and as uncle ben said: <b>`'With great power comes great responsibility'`</b>


## Pre - Running:
1. make sure you have git LFS (https://git-lfs.github.com/)
2. Clone Auto-Connect Repo
3. install requirements: `pip install -r requirements.txt`
4. Place your file in inputs sub-folder [`inputs/module_name`]. we support `ply`,`stl`.



## Running Auto connect:
run `code/main.py` using terminal followed by arguments:  
* <b>`--input, -in`</b> &nbsp; <span style="color:red"> required </span>  
   module name to use as input from inputs folder 
* <b>`--input-type, -it`</b> &nbsp; default= `ply`  
   module type <span style="color:green"> {STL, PLY}} </span>   
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
* <b>`--message, -m`</b>  
    a text to add to AutoConnect.txt output file (default: '')

## Example command Lines:
* `python .\code\main.py -in sandal -c 3 -m 'This Will Be Written In Terminal & outputs\sandal_3\AutoConnect.txt'`
* `python .\code\main.py -c 4 -in ball -i 5 -r 1`
* `python .\code\main.py -c 4 -in scissors -i 2 -r 1 -cv`
* `python .\code\main.py -c 4 5 -in cube -i 20 -r 2 -cv`
  


## Proposal of Software Mitigation for Non-Printable Outputs

Our goal is to add volume to the output mesh (holder), without modifying the contact of the holder with the object. we suggest here a naive approach.  
we mark the original holder (surface) as H and normal of the face f<sub>i</sub> as n<sub>i</sub>.  
1. defining the required 'width' of the holder:  
   marked by &delta;.  
   (can be passed through arguments or be hard-coded) 
2. Generating a shifted copy of H:  
   for each face f<sub>i</sub> in the holder we add a face g<sub>i</sub> that satisfies: g<sub>i</sub> = f<sub>i</sub> + &delta;n<sub>i</sub>.  
   making a duplicate of H shifted by the desired width, to a direction that will not effect the contact with the object.  we mark it H<sub>&delta;</sub>
3. connecting the parallel surfaces H & H<sub>&delta;</sub> by:  
   for each g<sub>i</sub> ( a face of H<sub>&delta;</sub> ), we connect each vertex to all the vertices f<sub>i</sub> (the matching fave in H).  
   this will also assure maintaining the triangles based mesh.
