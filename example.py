import numpy as np

def dist(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

def rotating_calipers(v):
    precision = 10000
    v *= precision
    n = len(v)
    A1, A2 = np.zeros((n, 2)), np.zeros((n, 2))
    ind = 0
    
    i, k, m = 0, 1, n - 1
    prev = dist(v[m], v[i], v[k])
    while (curr := dist(v[m], v[i], v[k+1])) > prev:
        prev = curr
        k += 1
        
    j = k
    while i <= k and j <= m:
        A1[ind], A2[ind] = v[i], v[j]
        ind += 1
        prev = dist(v[i], v[i+1], v[j])
        while j < m and (curr := dist(v[i], v[i+1], v[j+1])) > prev:
            prev = curr
            A1[ind], A2[ind] = v[i], v[j]
            ind += 1
            j += 1
        i += 1
    v /= precision
    return np.max(np.linalg.norm(A1 - A2, axis=1)) / precision