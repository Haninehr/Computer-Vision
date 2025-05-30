import math
import random
import numpy as np
import numpy.linalg as la
import scipy.ndimage as ndi
from scipy import signal

'''
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !!! NE MODIFIEZ PAS LE CODE EN DEHORS DES BLOCS TODO. !!!
 !!!  L'EVALUATEUR AUTOMATIQUE SERA TRES MECHANT AVEC  !!!
 !!!            VOUS SI VOUS LE FAITES !               !!!
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

# transformation coordonnées cartésiennes -> homogènes
pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])

# suppression de la dernière coordonnée
unpad = lambda x: x[:,:-1]


def eight_points(pts1, pts2):
    """
    TODO4.1
       Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)           
       [O] F, the fundamental matrix (3x3 matrix)
    """    
    
    assert (pts1.shape[0] == pts2.shape[0]),\
        'Nombre différent de points en pts1 et pts2'
    
    F = None    
    
    # TODO-BLOC-DEBUT    
    #---------------------------- Todo 4.1 -------------------------
    #Rouibah Hanine !
    def normaliser_pts(pts1,pts2):
        
        N = pts1.shape[0]
        N2 = pts2.shape[0]
        
        pts1_h =np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        pts2_h =np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    
        
   
        c = np.mean(pts1, axis=0) 
        c2 = np.mean(pts2, axis=0)

        
        
        centered = (pts1 - c)
        centered2= (pts2 - c2)
        
  
        ecart_type1 = np.sqrt((1/N)*(np.sum((centered**2)))) 
        ecart_type2 = np.sqrt((1/N2)*(np.sum((centered2**2)))) 
        
      
        s1 = np.sqrt(2) / ecart_type1

        s2 = np.sqrt(2) / ecart_type2
       
        T = np.array([
            [s1, 0, -s1 * c[0]],
            [0, s1, -s1 * c[1]],
            [0, 0, 1]
        ])
        T2 = np.array([
            [s2, 0, -s2 * c2[0]],
            [0, s2, -s2 * c2[1]],
            [0, 0, 1]
        ])
        
        
        points1_normalized = (T @ pts1_h.T).T
        points2_normalized = (T2 @ pts2_h.T).T
        
        return points1_normalized[:, :2], T , points2_normalized[:, :2], T2
    
    

    def rangF2(F):

        U,D,VT = np.linalg.svd(F)
        if np.linalg.det(F)==0:
          
            D_new = np.diag([(D[0]+D[1])/2, (D[1]+D[0])/2, 0])
        else:
            D_new = np.diag([D[0], D[1], 0])
        F_rang2 = U @ D_new @ VT
    
        return F_rang2

    
    if pts1.shape[0] >= 8:  
        pts1_n, Tr1, pts2_n, Tr2 = normaliser_pts(pts1,pts2)
      
        x1,y1 = pts1_n[:, 0], pts1_n[:, 1]
        x2,y2 = pts2_n[:, 0], pts2_n[:, 1]
        
        A = np.column_stack((
            x2 * x1,x2 * y1,x2,y2 * x1,y2 * y1,y2,x1,y1,np.ones(len(x1)) 
        ))
        
        _, _, V = np.linalg.svd(A)
        
        F_vec = V[-1]
    
        F_norm = F_vec.reshape(3, 3)
        F_norm = rangF2(F_norm)

        
        F = Tr2.T @ F_norm @ Tr1
        
    else:
        F=np.eye(3)
    
    #-------------------------------------- End -------------------------------------
    # TODO-BLOC-FIN

    return F


def ransac(keypoints1, keypoints2, matches, n_iters=500, threshold=1e-4):
    """
    TODO4.2
       RANSAC pour trouver une transformation projective robuste

       [I] keypoints1,  tableau M1 x 2, chaque rangée contient les coordonnées 
                        d'un point-clé (x_i,y_i) dans image1
           keypoints2,  tableau M2 x 2, chaque rangée contient les coordonnées 
                        d'un point-clé (x'_i,y'_i) dans image2
           matches,     tableau N x 2, chaque rangée représente une correspondance
                        [indice_dans_keypoints1, indice_dans_keypoints2]
           n_iters,     le nombre d'itérations à effectuer pour RANSAC
           threshold,   le seuil pour sélectionner des bonnes correspondances
       [O] F,           une estimation robuste de la matrice Fondamentale F
           goodmatches, tableau max_inliers x 2 contenant les indices des bonnes correspondances 
    """
    
    # Matrice Fondamentale
    F = None
    
    #indices des bonnes correspondances
    goodmatches = None
    
    # Initialisation du générateur de nombres aléatoires
    # fixé le seed pour pouvoir comparer le résultat retourné par 
    # cette fonction par rapport à la solution référence
    random.seed(131)
    
    # TODO-BLOC-DEBUT   
    #--------------------------------- Todo 4.2 -------------------------------
    #Rouibah Hanine !

    
    def compute_error(F, pt1, pt2):
    
      x1 = np.append(pt1, 1)
      x2 = np.append(pt2, 1)

      err_alg = np.abs(x2.T @ F @ x1)

      return err_alg
    
    
    goodmatches = []
    
    K1 =  keypoints1
    K2 = keypoints2
    
    for _ in range(n_iters):
        r = random.sample(range(len(matches)), 8)
        source_points = np.array([K1[matches[i][0]] for i in r])
        destination_points = np.array([K2[matches[i][1]] for i in r])
        F = eight_points(source_points, destination_points)
        
        inliers = []
      
        for i, match in enumerate(matches):
            source_point = np.array(K1[match[0]])
            destination_point = np.array(K2[match[1]])

            
            err= compute_error(F,source_point,destination_point)
            if   err < threshold :
              
                  inliers.append(i)
                 
        if len(inliers) > len(goodmatches):
            goodmatches = inliers
    
    print(len(goodmatches))
    if len(goodmatches) >= 0:
       
        source_points = np.array([K1[matches[i][0]] for i in goodmatches])
        destination_points = np.array([K2[matches[i][1]] for i in goodmatches])
        F = eight_points(source_points, destination_points)       
        goodmatches = [matches[i] for i in goodmatches]  
        
        
    goodmatches = np.array(goodmatches)
    
    #----------------------------- End ---------------------------------         
    # TODO-BLOC-FIN
                
    return F, goodmatches
        

def epipolar_match(im1, im2, F, pts1, W = 7):
    """
    TODO4.3
       Compute keypoints correspondences using Epipolar geometry
       [I] im1, image 1 (H1xW1x3 matrix)
           im2, image 2 (H2xW2x3 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
    """
    
    assert len(im1.shape) == 3 and len(im2.shape) == 3, '...'
    
    pts2 = None
    
    # TODO-BLOC-DEBUT  
    #---------------------------------- Todo 4.3 ---------------------------------
    
    #Rouibah Hanine !



    def SSD(w1,w2):
        score = np.sum((w1 - w2) ** 2)
        return score
    
    
    def normalization(w):
        
        if(np.var(w)<1e-10):
            w=0
        else:

            w = w - np.mean(w)  
            w = w / (np.std(w))  
        return w

    def extrair_patch(I,y,x,h_window):
       
        patch_size = 2 *h_window +1 
        patch = np.zeros((patch_size, patch_size,3), dtype=I.dtype)
        
      
        h,w,_ = I.shape
        
        

        y_s = int(y) - h_window
        y_patch_start = max(0,-y_s)
        y_img_start = max(0, y_s)

        y_end = int(y) + h_window +1
        y_patch_end = patch_size - max(0,y_end-h)
        y_img_end = min(h, y_end)


        x_s = int(x) - h_window
        x_patch_start = max(0, -x_s)
        x_img_start = max(0, x_s)

        x_end = int(x) + h_window +1
        x_patch_end = patch_size - max(0,x_end-w)
        x_img_end = min(w, x_end)
        
    
        
        patch[y_patch_start:y_patch_end, x_patch_start:x_patch_end, :] = I[y_img_start:y_img_end, x_img_start:x_img_end , :]
      
        return patch

    I1 = im1
    I2 = im2



    I1 = ndi.gaussian_filter(I1,sigma=5.0)
    I2 = ndi.gaussian_filter(I2, sigma=5.0)
    
    
    N = pts1.shape[0]  
    h,w,_  = I2.shape
    pts2 = np.zeros_like(pts1, dtype=np.float64)
    half_window = W // 2
    
    for i in range(N):
       
        pl = np.array([(int(pts1[i, 0])), int((pts1[i, 1])), 1])
        lr = F @ pl  
        a,b,c = lr
        
        nrm= np.sqrt(a**2 +b**2)
        if nrm > 1e-10:
            a /=nrm
            b /=nrm
            c /=nrm
        
       
        intersections = []
        if b !=0:
            y = -c /b
            if 0<= y < h:
                intersections.append((0,y))
            y = -(a *(w-1) + c) / b
            if 0<= y < h:
                intersections.append((w-1,y))
        if a != 0:
            x = -c/a
            if 0<= x< w:
                intersections.append((x,0))
            x = -(b *(h-1) + c) / a
            if 0<= x< w:
                intersections.append((x,h-1))
     
        if len(intersections) == 0:
          
            pts2[i] = [None,None]
         
            continue
        
        
        candidates = []
        if abs(b) > abs(a):  
            x_c = np.arange(0, w, 1)
            y_c = -(a *x_c + c) / b  
            valid = (y_c >= 0) & (y_c < h)
            candidates = np.vstack([x_c[valid], y_c[valid]]).T
        else:  
            y_c = np.arange(0,h, 1)
            x_c = -(b * y_c + c) / a  
            valid = (x_c >= 0) & (x_c < w)
            candidates = np.vstack([x_c[valid], y_c[valid]]).T
        
        if len(candidates) == 0:
            pts2[i] = [None, None]
            
            continue
        
        
        x_l, y_l = (pts1[i]).astype(int)
        
        if not (0 <= x_l < w and 0 <= y_l < h):
         
            pts2[i] = [None, None]  
           
            continue
        
        window_l = extrair_patch(I1, y_l,x_l,half_window)
        print((window_l.shape))
        window_l = normalization(window_l)
        
        
        best_score = float('inf')
        best_pc = None
        
        for pc in candidates:
            x_c,y_c = pc.astype(int)
            window_c = extrair_patch(I2,y_c,x_c,half_window)
            window_c = normalization(window_c)
            print(np.sum(window_c))
            score = SSD(window_l,window_c)  
   
            
            if score < best_score:
                best_score = score
                best_pc = pc
        
        if best_pc is not None:
           
            pts2[i] = best_pc
        else:
            pts2[i] = [None, None]  
         
    #-------------------------------------- End ---------------------------------------
    # TODO-BLOC-FIN
    
    return pts2


def estimate_camera_pose( F, principal_point, focal_distance, base_distance ):
    """
    TODO4.4
       Estimate the four possible camera poses
       [I] F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           principal_point, camera's principal point coordinates (1x2 tuple)
           focal_distance, camera's x and y focal lengths (1x2 tuple)
           base_distance, distance betwenn the origins of the cameras (scalar)                      
       [O] K, camera's intrinsic parameters (3x3 matrix)
           Rt_list, camera's extrinsic parameters [R|t] (list of four 4x3 matrices)  
    """
    
    K       = None
    Rt_list = None

    # TODO-BLOC-DEBUT    
    #----------------------------- Todo 4.4 --------------------------------------
    #Rouibah Hanine !

   
    cx,cy = principal_point 


    K = np.eye(3)
    K[:2,2] = cx,cy
    np.fill_diagonal(K[:2,:2], focal_distance)

  
    E = K.T @ F @ K   
    
    U, _, Vt = np.linalg.svd(E)
    
    
    S_new = np.diag([1, 1,  0])

    E = U @ S_new @ Vt


    U, _, Vt = np.linalg.svd(E)

 
    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    


    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    
    t = U[:, 2] 
   
    t1 = t
    t2 = -t
    
  

    t2 = (K) @ t2
    t2 = t2 / np.linalg.norm(t2) * (base_distance )

    t1 =  (K)  @ t1
    t1= t1 / np.linalg.norm(t1) * (base_distance )
    
    if np.linalg.det(R1) < 0:
        R1 = -R1
   
        
    if np.linalg.det(R2) < 0:
        R2 = -R2
        
 
    
    Rt_list = [
        
        np.hstack((R1, t1.reshape(-1, 1))),
        np.hstack((R1,  t2.reshape(-1, 1))),
        np.hstack((R2,t1.reshape(-1, 1))),
        np.hstack((R2, t2.reshape(-1, 1))),
           
    ]


    #------------------------------------- End --------------------------------
    # TODO-BLOC-FIN

    return K, Rt_list

    
def triangulate(P1, pts1, P2, pts2):
    """
    TODO4.5-1
       Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
    """    
    
    pts3d = None
    
    # TODO-BLOC-DEBUT    
    #----------------------- Todo 4.5-1 ------------------------
    #Rouibah Hanine !
    pts3d = []
    for pt1,pt2 in zip(pts1, pts2):
        x1,y1 = pt1
        x2,y2 = pt2

        D = np.array([
            x1 *P1[2] - P1[0],
            y1 *P1[2] - P1[1],
            x2 *P2[2] - P2[0],
            y2 *P2[2] - P2[1],
        ])

        _, _,VT = np.linalg.svd(D)
        X= VT[-1]
        X = X /X[3] 
        pts3d.append(X[:3])
    
    pts3d=np.array(pts3d)
    #---------------------------------- End --------------------------------
    # TODO-BLOC-FIN
    
    return pts3d


def check_chirality(K, Rt_list, pts1, pts2):
    """
    TODO4.5-2
       Chirality check
       [I] K, camera intrinsic matrix (3x3 matrix)
           Rt_list, camera's extrinsic parameters [R|t] (list of four 4x3 matrices)             
           pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] Rt, correct camera's extrinsic parameters [R|t] (4x3 matrices)             
           pts3d_list, 3D points in space (list of four Nx3 matrix)
    """    

    Rt = None
    pts3d_list  = None
    
    # TODO-BLOC-DEBUT    
    #------------------------------ Todo 4.5-2 --------------------------
    #Rouibah Hanine !
   
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  
    
    max_valid_count =-1
    best_Rt = None
    
    pts3d_list = []

    for i,Rt in enumerate(Rt_list):
        P2 = K @ Rt
        pts3d = triangulate(P1,pts1, P2, pts2)
        pts3d_list.append(pts3d)  

        R = Rt[:, :3]
        t = Rt[:, 3]

    
        count = 0
        for X in pts3d:
            X_cam1 = X   
            X_cam2 = R @ (X - t)
            if X_cam1[2] > 0 and X_cam2[2] > 0: 
                count += 1

        if count > max_valid_count:
            max_valid_count =count
            best_Rt =Rt
            
    Rt =  best_Rt

    #------------------------------------- End -------------------------------------
    # TODO-BLOC-FIN

    return Rt, pts3d_list

def compute_matching_homographies(F, principal_point, pts1, pts2):
    """
    TODO5
       Compute matching homography matrices     
       [I] F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           principal_point, camera's principal point coordinates (1x2 tuple)
           pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] H1, homography transformation matrix for the first image (3x3 matrix)             
           H2, homography transformation matrix for the second image (3x3 matrix)             
    """    

    H1 = None
    H2 = None
    
    # TODO-BLOC-DEBUT    
    #------------------------------- Todo 5 ---------------------------
    #Rouibah Hanine !
    
    def skew_sym(e):
    
        e = np.array(e)
        return np.array([
            [0,-e[2], e[1]],
            [e[2],0,    -e[0]],
            [-e[1], e[0],  0]
        ])
    

    def compute_epipole(F):
        _, _, Vt = np.linalg.svd(F)
        e = Vt[-1]
        e = e / e[-1] 
        return e

    def compute_H2(e2 , T):
         
        e2_p = T @ e2
        e2_p = e2_p / e2_p[2]
        e2x = e2_p[0]
        e2y = e2_p[1]

        
        if e2x >= 0:
            alpha = 1
        else:
            alpha = -1
        R1 = alpha* e2x / np.sqrt(e2x ** 2 + e2y ** 2)
        R2 = alpha * e2y / np.sqrt(e2x ** 2 + e2y ** 2)
        R = np.array([[R1, R2,  0], [-R2,R1,0], [0,0,1]])

        e2_p = R @ e2_p
        f = e2_p[0] 
      
        G = np.array([[1,0,0], [0, 1, 0], [-1/f, 0,1]])
        
        H2 = np.linalg.inv(T) @ G @ R @ T  
        return H2

    def compute_H1(e2, F,pts1,pts2, H2):
        
        e_x = skew_sym(e2)

        M = e_x @ F + e2.reshape(3,1) @ np.array([[1, 1, 1]])  

        points1_h = np.hstack([pts1,np.ones((pts1.shape[0], 1))])
        points2_h = np.hstack([pts2,np.ones((pts2.shape[0], 1))])

        points1_t = H2 @ M @ points1_h.T
        points2_t = H2 @ points2_h.T
        points1_t /= points1_t[2,:]
        points2_t /= points2_t[2,:]
        b = points2_t[0, :]
        a = np.linalg.lstsq(points1_t.T, b, rcond=None)[0]
        H_A = np.array([a,[0, 1, 0], [0, 0, 1]])

        H1 = H_A @ H2 @ M
        return H1
    

    cx,cy = principal_point 

  
    T = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ])


    e2 = compute_epipole(F.T)

    H2 = compute_H2(e2, T)

    H1 = compute_H1(e2, F,pts1 , pts2 , H2)

    #----------------------------- End -----------------------------
    # TODO-BLOC-FIN

    return H1, H2


def compute_disparity(im1, im2, max_disparity, win_size):
    """
    TODO6.1
       Calcul de la carte de disparité
       [I] im1, rectified image 1 (HxWx3 matrix)
           im2, rectified image 2 (HxWx3 matrix)           
           max_disparity, maximum disparity to check (scalar)
           win_size, windows size for block matching (scalar > 0)
       [O] disp, disparity map associated with im1 (HxW matrix)
    """    
    assert im1.shape[0] == im2.shape[0] and \
           im1.shape[1] == im2.shape[1], 'les images doivent avoir des dimensions identiques'
    
    assert 0 < max_disparity and max_disparity < im2.shape[1], 'max_disparity < im1.shape[1]'
    
    disp = None   

    # TODO-BLOC-DEBUT     
    #---------------------------------- Todo 6.1 ---------------------------
    #Rouibah Hanine !
   
    
    if im1.ndim == 3:
        im1 = np.mean(im1,axis=2)
    if im2.ndim == 3:
        im2 = np.mean(im2,axis=2)

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    h,w = im1.shape
    half_win = win_size//2
    mask = np.ones((win_size, win_size), dtype=np.float64)

   
    pad_shape = ((half_win, half_win), (half_win, half_win))
    im1_padded = np.pad(im1, pad_shape, mode='reflect')
    im2_padded = np.pad(im2, pad_shape, mode='reflect')

    disp = np.zeros((h,w), dtype=np.float64)
    min_cout = np.full((h,w), np.inf, dtype=np.float64)

    for d in range(max_disparity + 1):
        

        t_im2 = np.zeros_like(im2_padded)
        t_im2[:,d:w + 2 * half_win] = im2_padded[:, :w + 2 * half_win - d]  

      
        cout = signal.convolve2d((im1_padded - t_im2) ** 2, mask, mode='valid') 

        
        new_mask = cout< min_cout
        disp[new_mask] = d
        min_cout[new_mask] = cout[new_mask]
            
   
    #--------------------------- ENd -------------------------------
    # TODO-BLOC-FIN
        
    return disp

       
def cross_disparity(im1, im2, disp1, max_disparity, win_size):
    """
    TODO6.2
       Validation de la carte de disparité
       [I] im1, rectified image 1 (HxWx3 matrix)
           im2, rectified image 2 (HxWx3 matrix)           
           disp1, left disparity matrix (HxW matrix)
           max_disparity, maximum disparity to check (scalar)
           win_size, windows size for block matching (scalar > 0)
       [O] disp2, disparity map associated with im2 (HxW matrix)
           dispc, coherent disparity map for im1 (HxW)
    """    
    assert im1.shape[0] == im2.shape[0] and \
           im1.shape[1] == im2.shape[1], 'les images doivent avoir des dimensions identiques'
    
    assert 0 < max_disparity and max_disparity < im2.shape[1], 'max_disparity < im1.shape[1]'
    
    disp2 = None
    dispc = None

    # TODO-BLOC-DEBUT   
    # Rouibah Hanine !  
    pass
    #Bonus not implemented !
    # TODO-BLOC-FIN
    
    return disp2, dispc


def fill_holes(dispc):    
    """
    TODO6.3
       Disparity holes filling 
       [I] dispc, coherent disparity map with holes (negative values) (HxW)
       [O] dispf, filled disparity map (HxW)
    """    

    dispf = None
        
    # TODO-BLOC-DEBUT 
    # Rouibah Hanine !    
    pass
   
    #fill_left
    #fill_right 




    #End of the Line !
    # TODO-BLOC-FIN

    return dispf

