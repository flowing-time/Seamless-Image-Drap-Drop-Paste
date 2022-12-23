import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
import heapq

import sys
sys.setrecursionlimit(20000)

import ddp


# # Test Tahoe boat
# Read images and mask
source = cv2.imread('images/tahoe/boat2.jpg')
target = cv2.imread('images/tahoe/water.jpg')
mask0 = cv2.imread('images/tahoe/mask_boat2.png', cv2.IMREAD_GRAYSCALE)
p = (180, 320)

mask0[mask0<255] = 0
normal_clone = cv2.seamlessClone(source, target, mask0, p, cv2.NORMAL_CLONE)
cv2.imwrite("images/output/boat_default2.jpg", normal_clone)

blend = ddp.clone(source, mask0, target, p)
cv2.imwrite("images/output/boat_ddp2.jpg", blend)




# # Test Yosemite tree
# Read images and mask
source = cv2.imread('images/Yosemite/tree.jpg')
target = cv2.imread('images/Yosemite/valley.jpg')
mask0 = cv2.imread('images/Yosemite/mask_tree.png', cv2.IMREAD_GRAYSCALE)
p = (700, 250)

mask0[mask0<255] = 0
normal_clone = cv2.seamlessClone(source, target, mask0, p, cv2.NORMAL_CLONE)
cv2.imwrite("images/output/tree_default.jpg", normal_clone)

blend = ddp.clone2(source, mask0, target, p, 30)
cv2.imwrite("images/output/tree_ddp.jpg", blend)




# # Test dolphin image
# Read images and mask
source = cv2.imread('images/tank/dolphin2.jpg')
target = cv2.imread('images/tank/dolphin.jpg')
mask0 = cv2.imread('images/tank/mask_dolphin.png', cv2.IMREAD_GRAYSCALE)
p = (330, 400)


smask = np.where(mask0<255, 0, 255).astype(np.uint8)
normal_clone = cv2.seamlessClone(source, target, smask, p, cv2.NORMAL_CLONE)
cv2.imwrite("images/output/dolphin_default.jpg", normal_clone)

blend = ddp.clone2(source, mask0, target, p, 10)
cv2.imwrite("images/output/dolphin_ddp.jpg", blend)