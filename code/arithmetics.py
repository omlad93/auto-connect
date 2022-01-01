import numpy as np


epsilon     = 10**-5
size        = lambda v : np.sqrt(np.sum(np.dot(v,v)))
zero_clip   = lambda x : max(x,epsilon)
points_diff = lambda p1,p2 : [p1[i] - p2[i] for i in range(len(p1))]
points_dist = lambda p1,p2 : np.sqrt(np.sum(np.square(points_diff(p1,p2)))) 
rad_to_deg  = lambda rad: 360*rad/(2*np.pi)

def angle(v1,v2):
    dot = np.dot(v1,v2)/zero_clip(size(v1)*size(v2))
    assert type(dot)==np.float64, f'Angle: type(dot)={type(dot)}\b       type(v1)={type(v1)}\n       type(v2)={type(v2)}'
    angle = np.arccos(dot)
    return angle

def distance_from_plane(point:np.ndarray, plane: np.ndarray) -> float:
    ''' 
        A function to find minimal distance of point from plane
        point = [px, py, pz]
        plane = [a, b, c, d] 
    '''
    x,y,z = point
    a,b,c,d = plane
    numerator = abs (a*x + b*y + c*z + d)
    denominator = size(np.array((a,b,c)))
    return numerator/denominator if (denominator != 0) else np.inf