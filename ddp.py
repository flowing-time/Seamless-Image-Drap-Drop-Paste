# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt


# %%
import collections
import heapq


# %%
def clone(source, mask0, target, p):
    # source and mask are in same shape. p is a point on target.
    mask0 = np.where(mask0<255, False, True)

    # Get source clip and target clip
    s_tlx, s_tly, s_brx, s_bry = getTLBR(mask0)
    s_clip = source[s_tly:s_bry, s_tlx:s_brx]
    h, w = s_clip.shape[:2]
    t_tlx, t_tly = p[0] - w//2, p[1] - h//2
    t_brx, t_bry = t_tlx + w, t_tly + h
    t_clip = target[t_tly:t_bry, t_tlx:t_brx]

    # mask and mask_obj are local mask on s_clip, t_clip and diff
    diff = np.sum((t_clip.astype(np.float) - s_clip.astype(np.float)) ** 2, axis=2) ** 0.5
    mask = mask0[s_tly:s_bry, s_tlx:s_brx]
    mask_obj = getObjMask(s_clip, mask)
    tlbr = getTLBR(mask_obj)
    cut_row = (tlbr[1] + tlbr[3]) // 2
    j_obj = np.nonzero(mask_obj[cut_row])[0][0]

    bdry = getBoundary_dfs(mask)
    E, k = bdryEngy(diff, mask)
    
    while True:
        nmask, nbdry = getNewMask(diff, mask, bdry, k, mask_obj, cut_row, j_obj)
        nE, nk = bdryEngy(diff, nbdry)
        if nE >= E:
            break
        mask, bdry, E, k = nmask, nbdry, nE, nk

    d_tlx, d_tly, d_brx, d_bry = getTLBR(mask)
    new_p = t_tlx + (d_tlx + d_brx)//2, t_tly + (d_tly + d_bry)//2
    
    mask = mask.astype(np.uint8) * 255
    blend = cv2.seamlessClone(s_clip, target, mask, new_p, cv2.NORMAL_CLONE)

    return blend


# %%
def getTLBR(mask):
    rows, cols = np.nonzero(mask)
    return cols.min(), rows.min(), cols.max()+1, rows.max()+1


# %%
def getObjMask(image, mask):
    gmask = np.zeros_like(mask, dtype=np.uint8)
    gmask[mask] = cv2.GC_PR_FGD
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # rect is None
    cv2.grabCut(image, gmask, None, bgdModel, fgdModel,5, cv2.GC_INIT_WITH_MASK)
    gmask = np.where((gmask==2)|(gmask==0), False, True)
    getMajor_dfs(gmask)

    return gmask


# %%
def getNewMask(diff, mask, bdry, k, mask_obj, cut_row, j_obj):
    energy = (diff - k) ** 2
    j0 = np.nonzero(mask[cut_row])[0][0] 
    C = set( (cut_row, j) for j in range(j0, j_obj) )
    rmask = (mask ^ mask_obj) | bdry
    
    minE = float('inf')
    for src_p in sorted(C):
        tgt_p = src_p[0] + 1, src_p[1]
        if not rmask[tgt_p]:
            continue
        #print(src_p, tgt_p)
        E, path = minCost(energy, rmask, src_p, tgt_p, C)
        if E < minE:
            minE = E
            nbdry = path
        
    nmask = fillBoundary_bfs(nbdry, mask_obj)
    
    return nmask, nbdry


# %%
def minCost(engy, rmask, src_p, tgt_p, C):
    # Dijkstra: to find the min cost from source pixel (i, j) to (i+1, j)
    # source is staring point, C is the cut line of the ring. The path cannot cross C.
    # rmask is the mask of ring. The path must be inside rmask.
    # tgt_p = (src_p[0] + 1, src_p[1])
    r, c = engy.shape # same shape as rmask
    visited = np.zeros((r, c), dtype=np.bool)
    cost = np.full((r, c), float('inf'))
    prev = np.zeros((r, c, 2), dtype=np.int64)
    
    cost[src_p] = engy[src_p]
    pq = []
    heapq.heappush(pq, (cost[src_p], src_p))
    
    while pq:
        u = heapq.heappop(pq)[1]
        if visited[u]:
            continue
        visited[u] = True
        u_nbh = nbh_src(*u) if u in C else nbh1(*u)
        u_nbh = filter(lambda u:0<=u[0]<r and 0<=u[1]<c, u_nbh)
        
        for v in u_nbh:
            if visited[v] or (not rmask[v]):
                continue
            alt = cost[u] + engy[v]
            if alt < cost[v]:
                cost[v] = alt
                heapq.heappush(pq, (cost[v], v))
                prev[v] = u
            
    path = np.zeros_like(rmask, dtype=np.bool)
    p = tgt_p
    while True:
        path[p] = True
        if p == src_p:
            break
        p = tuple(prev[p])
    
    return cost[tgt_p], path


# %%
def bdryEngy(diff, bdry):
    # diff dtype=float
    #bdry = getBoundary_dfs(mask)
    k = diff[bdry].sum() / bdry.sum()
    E = np.sum( (diff[bdry] - k) ** 2 )
    
    return E, k


# %%
#def getBoundary(mask):
#    r, c = mask.shape
#    bdry = np.zeros((r, c), dtype=np.bool)
#    for i in range(r):
#        for j in range(c):
#            if not mask[i, j]:
#                continue
#            bdry[i, j] = (i in (0, r-1)) or (j in (0, c-1)) or any(not mask[p] for p in nbh2(i, j))
#
#    return bdry


# %%
def getBoundary_dfs(mask):
    # only get one boundary
    r, c = mask.shape
    bdry = np.zeros((r, c), dtype=np.bool)

    def dfs(i, j):
        bdry[i, j] = True
        for x, y in nbh1(i, j):
            if 0 <= x < r and 0 <= y < c and mask[x, y] and (not bdry[x, y]):
                if (x in (0, r-1)) or (y in (0, c-1)) or any(not mask[p] for p in nbh2(x, y)):
                    dfs(x, y)
                    return

    for i in range(r):
        for j in range(c):
            if mask[i, j]:
                dfs(i, j)
                return bdry   


# %%
def fillBoundary(bdry):
    # only use bdry
    mask = bdry.copy()
    for i in range(bdry.shape[0]):
        if np.any(bdry[i]):
            lx, rx = np.nonzero(bdry[i])[0][[0, -1]]
            mask[i, lx:rx] = True

    return mask


# %%
def fillBoundary_bfs(bdry, mask_obj):
    r, c = bdry.shape
    mask = bdry | mask_obj
    q = collections.deque(zip(*np.nonzero(mask_obj)))
    while q:
        i, j = q.popleft()
        if bdry[i, j]:
            continue
        for x, y in nbh2(i, j):
            if 0<=x<r and 0<=y<c and (not mask[x, y]):
                mask[x, y] = True
                q.append((x, y))
        

    return mask


# %%
def nbh1(i, j):
    return (i-1, j), (i+1, j), (i, j-1), (i, j+1)


# %%
def nbh2(i, j):
    return (i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j-1), (i+1, j-1), (i-1, j+1), (i+1, j+1)


# %%
def nbh_src(i, j):
    return (i-1, j), (i, j-1), (i, j+1)

# %%
# Expand the mask_obj to get Phi
def expand_bfs(mask, w):
    r, c = mask.shape
    nmask = mask.copy()
    alpha = np.zeros_like(mask, dtype=np.float)
    alpha[mask] = 1.0

    q = collections.deque( (p, 0) for p in zip(*np.nonzero(nmask)) )
    while q:
        p, level = q.popleft()
        if level < w:
            nlevel = level + 1
            nalpha = 1.0 - nlevel / w 
            for x, y in nbh2(*p):
                if 0<=x<r and 0<=y<c and (not nmask[x, y]):
                    nmask[x, y] = True
                    alpha[x, y] = nalpha
                    q.append( ((x, y), nlevel) )

    return nmask, alpha


# %%
def getTargetMask(mask, target, p):
    tmask = np.zeros(target.shape[:2], dtype=np.bool)
    s_tlx, s_tly, s_brx, s_bry = getTLBR(mask)
    h, w = s_bry - s_tly, s_brx - s_tlx
    t_tlx, t_tly = p[0] - w//2, p[1] - h//2
    t_brx, t_bry = t_tlx + w, t_tly + h
    tmask[t_tly:t_bry, t_tlx:t_brx] = mask[s_tly:s_bry, s_tlx:s_brx]
    return tmask


# %%
def clone_debug(source, mask0, target, p):
    # source and mask are in same shape. p is a point on target.
    mask0 = np.where(mask0<255, False, True)

    # Get source clip and target clip
    s_tlx, s_tly, s_brx, s_bry = getTLBR(mask0)
    s_clip = source[s_tly:s_bry, s_tlx:s_brx]
    h, w = s_clip.shape[:2]
    t_tlx, t_tly = p[0] - w//2, p[1] - h//2
    t_brx, t_bry = t_tlx + w, t_tly + h
    t_clip = target[t_tly:t_bry, t_tlx:t_brx]

    # mask and mask_obj are local mask on s_clip, t_clip and diff
    diff = np.sum((t_clip.astype(np.float) - s_clip.astype(np.float)) ** 2, axis=2) ** 0.5
    mask = mask0[s_tly:s_bry, s_tlx:s_brx]
    mask_obj = getObjMask(s_clip, mask)
    tlbr = getTLBR(mask_obj)
    cut_row = (tlbr[1] + tlbr[3]) // 2
    j_obj = np.nonzero(mask_obj[cut_row])[0][0]

    bdry = getBoundary_dfs(mask)
    E, k = bdryEngy(diff, mask)
    
    while True:
        nmask, nbdry = getNewMask(diff, mask, bdry, k, mask_obj, cut_row, j_obj)
        nE, nk = bdryEngy(diff, nbdry)
        if nE >= E:
            break
        mask, bdry, E, k = nmask, nbdry, nE, nk

    d_tlx, d_tly, d_brx, d_bry = getTLBR(mask)
    new_p = t_tlx + (d_tlx + d_brx)//2, t_tly + (d_tly + d_bry)//2
    
    #mask_s_clip = s_clip * mask[:, :, np.newaxis]
    mask = mask.astype(np.uint8) * 255
    blend = cv2.seamlessClone(s_clip, target, mask, new_p, cv2.NORMAL_CLONE)

    tmask0 = getTargetMask(mask0, target, p)
    tbdry0 = getBoundary_dfs(tmask0)

    tmask = getTargetMask(mask, target, new_p)
    tbdry = getBoundary_dfs(tmask)

    tmask_obj = getTargetMask(mask_obj, target, new_p)
    tbdry_obj = getBoundary_dfs(tmask_obj)

    blend_debug = blend.copy()
    blend_debug[tbdry0] = (0, 0, 255)
    blend_debug[tbdry_obj] = (255, 0, 0)
    blend_debug[tbdry] = (0, 255, 0)

    return blend, blend_debug



# %%
def clone1(source, mask0, target, p, expansion):
    # source and mask are in same shape. p is a point on target.
    mask0 = np.where(mask0<255, False, True)

    # Get source clip and target clip
    s_tlx, s_tly, s_brx, s_bry = getTLBR(mask0)
    s_clip = source[s_tly:s_bry, s_tlx:s_brx]
    h, w = s_clip.shape[:2]
    t_tlx, t_tly = p[0] - w//2, p[1] - h//2
    t_brx, t_bry = t_tlx + w, t_tly + h
    t_clip = target[t_tly:t_bry, t_tlx:t_brx]

    # mask and mask_obj are local mask on s_clip, t_clip and diff
    diff = np.sum((t_clip.astype(np.float) - s_clip.astype(np.float)) ** 2, axis=2) ** 0.5
    mask = mask0[s_tly:s_bry, s_tlx:s_brx]

    mask_obj = getObjMask(s_clip, mask)
    mask_obj = expand_bfs(mask_obj, expansion)[0]

    tlbr = getTLBR(mask_obj)
    cut_row = (tlbr[1] + tlbr[3]) // 2
    j_obj = np.nonzero(mask_obj[cut_row])[0][0]

    bdry = getBoundary_dfs(mask)
    E, k = bdryEngy(diff, mask)
    
    while True:
        nmask, nbdry = getNewMask(diff, mask, bdry, k, mask_obj, cut_row, j_obj)
        nE, nk = bdryEngy(diff, nbdry)
        if nE >= E:
            break
        mask, bdry, E, k = nmask, nbdry, nE, nk

    d_tlx, d_tly, d_brx, d_bry = getTLBR(mask)
    new_p = t_tlx + (d_tlx + d_brx)//2, t_tly + (d_tly + d_bry)//2
    
    #mask_s_clip = s_clip * mask[:, :, np.newaxis]
    mask = mask.astype(np.uint8) * 255
    blend = cv2.seamlessClone(s_clip, target, mask, new_p, cv2.NORMAL_CLONE)

    tmask0 = getTargetMask(mask0, target, p)
    tbdry0 = getBoundary_dfs(tmask0)

    tmask = getTargetMask(mask, target, new_p)
    tbdry = getBoundary_dfs(tmask)

    tmask_obj = getTargetMask(mask_obj, target, new_p)
    tbdry_obj = getBoundary_dfs(tmask_obj)

    blend_debug = blend.copy()
    blend_debug[tbdry0] = (0, 0, 255)
    blend_debug[tbdry_obj] = (255, 0, 0)
    blend_debug[tbdry] = (0, 255, 0)

    return blend, blend_debug



# %%
def clone2(source, mask0, target, p, expansion):
    # source and mask are in same shape. p is a point on target.
    mask0 = np.where(mask0<255, False, True)

    # Get source clip and target clip
    s_tlx, s_tly, s_brx, s_bry = getTLBR(mask0)
    s_clip = source[s_tly:s_bry, s_tlx:s_brx]
    h, w = s_clip.shape[:2]
    t_tlx, t_tly = p[0] - w//2, p[1] - h//2
    t_brx, t_bry = t_tlx + w, t_tly + h
    t_clip = target[t_tly:t_bry, t_tlx:t_brx]

    # mask and mask_obj are local mask on s_clip, t_clip and diff
    diff = np.sum((t_clip.astype(np.float) - s_clip.astype(np.float)) ** 2, axis=2) ** 0.5
    mask = mask0[s_tly:s_bry, s_tlx:s_brx]

    mask_obj = getObjMask(s_clip, mask)

    tlbr = getTLBR(mask_obj)
    cut_row = (tlbr[1] + tlbr[3]) // 2
    j_obj = np.nonzero(mask_obj[cut_row])[0][0]

    bdry = getBoundary_dfs(mask)
    E, k = bdryEngy(diff, mask)
    
    while True:
        nmask, nbdry = getNewMask(diff, mask, bdry, k, mask_obj, cut_row, j_obj)
        nE, nk = bdryEngy(diff, nbdry)
        if nE >= E:
            break
        mask, bdry, E, k = nmask, nbdry, nE, nk

    mask |= expand_bfs(mask_obj, expansion)[0]
    d_tlx, d_tly, d_brx, d_bry = getTLBR(mask)
    new_p = t_tlx + (d_tlx + d_brx)//2, t_tly + (d_tly + d_bry)//2
    
    #mask_s_clip = s_clip * mask[:, :, np.newaxis]
    mask = mask.astype(np.uint8) * 255
    blend = cv2.seamlessClone(s_clip, target, mask, new_p, cv2.NORMAL_CLONE)

    #tmask0 = getTargetMask(mask0, target, p)
    #tbdry0 = getBoundary_dfs(tmask0)

    #tmask = getTargetMask(mask, target, new_p)
    #tbdry = getBoundary_dfs(tmask)

    #tmask_obj = getTargetMask(mask_obj, target, new_p)
    #tbdry_obj = getBoundary_dfs(tmask_obj)

    #blend_debug = blend.copy()
    #blend_debug[tbdry0] = (0, 0, 255)
    #blend_debug[tbdry_obj] = (255, 0, 0)
    #blend_debug[tbdry] = (0, 255, 0)

    return blend




# %%
def clone3(source, mask0, target, p, expansion):
    # source and mask are in same shape. p is a point on target.
    # expansion is the expand width of mask_obj
    mask0 = np.where(mask0<255, False, True)

    # Get source clip and target clip
    s_tlx, s_tly, s_brx, s_bry = getTLBR(mask0)
    s_clip = source[s_tly:s_bry, s_tlx:s_brx]
    h, w = s_clip.shape[:2]
    t_tlx, t_tly = p[0] - w//2, p[1] - h//2
    t_brx, t_bry = t_tlx + w, t_tly + h
    t_clip = target[t_tly:t_bry, t_tlx:t_brx]

    # mask and mask_obj are local mask on s_clip, t_clip and diff
    diff = np.sum((t_clip.astype(np.float) - s_clip.astype(np.float)) ** 2, axis=2) ** 0.5
    mask = mask0[s_tly:s_bry, s_tlx:s_brx]

    mask_obj = getObjMask(s_clip, mask)

    tlbr = getTLBR(mask_obj)
    cut_row = (tlbr[1] + tlbr[3]) // 2
    j_obj = np.nonzero(mask_obj[cut_row])[0][0]

    bdry = getBoundary_dfs(mask)
    E, k = bdryEngy(diff, mask)
    
    while True:
        nmask, nbdry = getNewMask(diff, mask, bdry, k, mask_obj, cut_row, j_obj)
        nE, nk = bdryEngy(diff, nbdry)
        if nE >= E:
            break
        mask, bdry, E, k = nmask, nbdry, nE, nk

    nmask_obj, alpha = expand_bfs(mask_obj, expansion)
    phi = nmask_obj ^ mask_obj
    #cv2.imwrite("images/flower/phi.png", phi.astype(np.uint8)*255)
    nmask = mask | nmask_obj
    
    d_tlx, d_tly, d_brx, d_bry = getTLBR(nmask)
    new_p = t_tlx + (d_tlx + d_brx)//2, t_tly + (d_tly + d_bry)//2

    #cv2.imwrite("images/flower/phi.png", phi.astype(np.uint8)*255)
    #cv2.imwrite("images/flower/alpha.png", alpha * 255)
    mixture = getMixRegion_bfs(bdry, phi, expansion)
    #alpha[~mixture] = 1
    cv2.imwrite("images/flower/alpha.png", alpha * 255)
    alpha = alpha[:, :, np.newaxis]
    cv2.imwrite("images/flower/s_clip.jpg", s_clip)
    ns_clip = (alpha * s_clip + (1 - alpha) * t_clip).astype(np.uint8)
    #ns_clip[getBoundary_dfs(nmask)] = (0, 0, 255)
    cv2.imwrite("images/flower/s_clip_mix.jpg", ns_clip)


    nmask = nmask.astype(np.uint8) * 255
    blend = cv2.seamlessClone(ns_clip, target, nmask, new_p, cv2.NORMAL_CLONE)

    tmask0 = getTargetMask(mask0, target, p)
    tbdry0 = getBoundary_dfs(tmask0)

    tmask = getTargetMask(nmask, target, new_p)
    tbdry = getBoundary_dfs(tmask)

    tmask_obj = getTargetMask(mask_obj, target, new_p)
    tbdry_obj = getBoundary_dfs(tmask_obj)

    blend_debug = blend.copy()
    blend_debug[tbdry0] = (0, 0, 255)
    blend_debug[tbdry_obj] = (255, 0, 0)
    blend_debug[tbdry] = (0, 255, 0)

    return blend, blend_debug

# %%
def getMixRegion_bfs(bdry, phi, w):
    r, c = bdry.shape
    mixture = bdry & phi
    q = collections.deque( (p, 0) for p in zip(*np.nonzero(mixture)) )
    while q:
        p, level = q.popleft()
        if level < w:
            for x, y in nbh2(*p):
                if 0<=x<r and 0<=y<c and phi[x, y] and (not mixture[x, y]):
                    mixture[x, y] = True
                    q.append( ((x, y), level + 1) )

    return mixture




# %%
def getMajor_dfs(mask):
    r, c = mask.shape
    major = np.zeros((r, c), dtype=np.bool)
    max_size = 0

    def dfs(i, j):
        mask[i, j] = False
        visited[i, j] = True
        total = 1
        for x, y in nbh1(i, j):
            if 0 <= x < r and 0 <= y < c and mask[x, y]:
                total += dfs(x, y)

        return total
    

    for i in range(r):
        for j in range(c):
            if mask[i, j]:
                visited = np.zeros((r, c), dtype=np.bool)
                size = dfs(i, j)
                if size > max_size:
                    max_size = size
                    major = visited.copy()
    
    mask[major] = True
    return
                