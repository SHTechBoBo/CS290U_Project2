import os
import cv2
import sys
import time
import omniglue
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from omniglue import utils

imgs_dir = './imgs'

og = omniglue.OmniGlue(
    og_export="./models/og_export",
    sp_export="./models/sp_v6",
    dino_export="./models/dinov2_vitb14_pretrain.pth",
)

for i in range(10):
    # ===================== load images =====================
    img_a_path = f'{imgs_dir}/{i}a.png'
    img_b_path = f'{imgs_dir}/{i}b.png'
    img_a = np.array(Image.open(img_a_path).convert("RGB"))
    img_b = np.array(Image.open(img_b_path).convert("RGB"))
    match_kp_a, match_kp_b, match_confidences = og.FindMatches(img_a, img_b)
    num_matches = match_kp_a.shape[0]


    # ===================== filter matches =====================
    match_threshold = 0.02
    keep_idx = []
    for j in range(match_kp_a.shape[0]):
        if match_confidences[j] > match_threshold:
            keep_idx.append(j)
    num_filtered_matches = len(keep_idx)
    match_kp_a = match_kp_a[keep_idx]
    match_kp_b = match_kp_b[keep_idx]
    match_confidences = match_confidences[keep_idx]

    # ===================== visualize matches =====================
    viz = utils.visualize_matches(
      img_a,
      img_b,
      match_kp_a,
      match_kp_b,
      np.eye(num_filtered_matches),
      show_keypoints=True,
      highlight_unmatched=True,
      title=f"{num_filtered_matches} matches",
      line_width=2,
    )
    plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
    plt.axis("off")
    plt.imshow(viz)
    plt.imsave("./results/{}.png".format(i), viz)
    
    # ===================== compute relative pose =====================
    F, mask = cv2.findFundamentalMat(match_kp_a, match_kp_b, method=cv2.RANSAC)
    inner_kp_a = match_kp_a[mask.ravel() == 1]
    inner_kp_b = match_kp_b[mask.ravel() == 1]
    num_filtered_matches=len(inner_kp_a)

    filename = "./results/{}_F.npz".format(i)
    np.savez(filename, inlier_kp0=inner_kp_a, inlier_kp1=inner_kp_b)
    
    K = np.load('./K.npz')["K"]
    E = np.dot(K.T, np.dot(F, K))
    _, R, t, _ = cv2.recoverPose(E, inner_kp_a, inner_kp_b, K)
    print("Rotation Matrix (R):\n", R)
    print("Translation Vector (t):\n", t)

