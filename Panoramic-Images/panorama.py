import math
import numpy as np
import random

from utils import pad, unpad

'''
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !!! NE MODIFIEZ PAS LE CODE EN DEHORS DES BLOCS TODO. !!!
 !!!  L'EVALUATEUR AUTOMATIQUE SERA TRES MECHANT AVEC  !!!
 !!!            VOUS SI VOUS LE FAITES !               !!!
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

def fit_transform_matrix(p0, p1):
    """ Calcul la matrice de transformation H tel que p0 * H.T = p1

    Indication importante :
        Vous pouvez utiliser la fonction "np.linalg.lstsq" ou
        la fonction "np.linalg.svd" pour résoudre le problème.

    Entrées :
        p0 : un tableau numpy de dimension (M, 2) contenant
             les coordonnées des points à transformer
        p1 : un tableau numpy de dimension (M, 2) contenant
             les coordonnées des points destination

             Chaque coordonnée [x,y] dans p0 ou p1 indique 
             la position d'un point-clé [col, ligne] dans 
             l'image associée. C-à-d.  p0[i,:] = [x_i, y_i] 
             et  p1[j,:] = [x'_j, y'_j]

    Sortie :
        H  : Tableau numpy de dimension (3,3) représentant la 
             matrice de transformation d'homographie.
    """

    assert (p1.shape[0] == p0.shape[0]),\
        'Nombre différent de points en p1 et p2'

    H = None
    
    #TODO 1 : Calculez la matrice de transformation H.
    # TODO-BLOC-DEBUT    
    #Rouibah's !
    
    p0_p1 = np.concatenate((p0, p1), axis=1) 
    H = []
 
    for x0,y0,x1,y1 in p0_p1:
        H.append([-x0, -y0, -1, 0, 0, 0, x0*x1, y0*x1, x1])
        H.append([0, 0, 0, -x0, -y0, -1, x0*y1, y0*y1, y1])

    try:
        _, _, V = np.linalg.svd(np.array(H)) 
        
        H = V[-1,:].reshape(3, 3)
        
    except np.linalg.LinAlgError:
        H = np.eye(3)

   
    #H.R
    # TODO-BLOC-FIN

    return H

def ransac(keypoints1, keypoints2, matches, n_iters=500, threshold=1):
    """
    Utilisez RANSAC pour trouver une transformation projective robuste

        1. Sélectionnez un ensemble aléatoire de correspondances
        2. Calculez la matrice de transformation H
        3. Calculez les bonnes correspondances (inliers)
        4. Gardez le plus grand ensemble de bonnes correspondances
        5. En final, recalculez la matrice de transformation H sur 
           tout l'ensemble des bonnes correspondances

    Entrées :
        keypoints1 -- matrice M1 x 2, chaque rangée contient les coordonnées 
                      d'un point-clé (x_i,y_i) dans image1
        keypoints2 -- matrice M2 x 2, chaque rangée contient les coordonnées 
                      d'un point-clé (x'_i,y'_i) dans image2
        matches  -- matrice N x 2, chaque rangée représente une correspondance
                    [indice_dans_keypoints1, indice_dans_keypoints2]
        n_iters -- le nombre d'itérations à effectuer pour RANSAC
        threshold -- le seuil pour sélectionner des bonnes correspondances

    Sorties :
        H -- une estimation robuste de la matrice de transformation H
        matches[max_inliers] -- matrice (max_inliers x 2) des bonnes correspondances 
    """
    # indices des bonnes correspondances dans le tableau 'matches' 
    max_inliers = []
    
    # matrice de transformation Homographique
    H = None
    
    # Initialisation du générateur de nombres aléatoires
    # fixé le seed pour pouvoir comparer le résultat retourné par 
    # cette fonction par rapport à la solution référence
    random.seed(131)
    
    #TODO 2 : Implémentez ici la méthode RANSAC pour trouver une transformation robuste
    # entre deux images image1 et image2.
    # TODO-BLOC-DEBUT    
    
    itr = n_iters
    m=matches
    K1 = keypoints1
    K2 = keypoints2
    for _ in range(itr):
        r = random.sample(range(len(m)), 4)
        source_points = np.array([K1[m[i][0]] for i in r])
        destination_points = np.array([K2[m[i][1]] for i in r])
        H = fit_transform_matrix(source_points, destination_points)
    
        inliers = []
        for i, match in enumerate(m):
            source_point = np.array(K1[match[0]])
            destination_point = np.array(K2[match[1]])
            p = np.dot(H, np.append(source_point, 1))
            p /= p[2]  # Normalize !
            distance = np.linalg.norm(p[:2] - destination_point)
            if distance < threshold:
                inliers.append(i)
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
    
    source_points = np.array([K1[m[i][0]] for i in max_inliers])
    destination_points = np.array([K2[m[i][1]] for i in max_inliers])
    H = fit_transform_matrix(source_points, destination_points)
    
    #H.R
    
    # TODO-BLOC-FIN
    
    return H, matches[max_inliers]


def get_output_space(imgs, transforms):
    """
    Ceci est une fonction auxiliaire qui prend en entrée une liste d'images et
    des transformations associées et calcule en sortie le cadre englobant
    les images transformées.

    Entrées :
        imgs -- liste des images à transformer
        transforms -- liste des matrices de transformation.

    Sorties :
        output_shape (tuple) -- cadre englobant les images transformées.
        offset -- un tableau numpy contenant les coordonnées du coin (0,0) du cadre
    """

    assert (len(imgs) == len(transforms)),\
        'le nombre d\'images et le nombre de transformations associées ne concordent pas'

    output_shape = None
    offset = None

    # liste pour récupérer les coordonnées de tous les coins dans toutes les images
    all_corners = []

    for img, H in zip(imgs, transforms):
        # coordonnées du coin organisées en (x,y)
        r, c, _ = img.shape        
        corners = np.array([[0, 0], [0, r], [c, 0], [c, r]])

        # transformation homographique des coins          
        warped_corners = pad(corners.astype(float)).dot(H.T).T        
        all_corners.append( unpad( np.divide(warped_corners, warped_corners[2,:] ).T ) )
                          
    # Trouver l'étendue des cadres transformées
    # La forme globale du cadre sera max - min
    all_corners = np.vstack(all_corners)

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    
    # dimension (largeur, longueur) de la zone d'affichage retournée
    output_shape = corner_max - corner_min
    
    # Conversion en nombres entiers avec np.ceil et dtype
    output_shape = tuple( np.ceil(output_shape).astype(int) )
    
    # Calcul de l'offset (horz, vert) du coin inférieur du cadre par rapport à l'origine (0,0).
    offset = corner_min

    return output_shape, offset


def warp_image(img, H, output_shape, offset, method=None):
    """
    Déforme l'image img grace à la transformation H. L'image déformée
    est copiée dans une image cible de dimensions 'output_shape'.

    Cette fonction calcule également les coefficients alpha de l'image
    déformée pour un fusionnement ultérieur avec d'autres images.

    Entrées :
        img -- l'image à transformée
        H -- matrice de transformation
        output_shape -- dimensions (largeur, hauteur) de l'image transformée 
        offset --  position (horz, vert) du coin du cadre transformé.
        method -- paramètre de sélection de la méthode de calcul des coefficients alpha.
                  'hlinear' -- le alpha varie linéairement de 1.0 à 0.0
                              en horizontal à partir du centre jusqu'au
                              bord de l'image
                  'vlinear' -- le alpha varie linéairement de 1.0 à 0.0
                              en vertical à partir du centre jusqu'au
                              bord de l'image
                  'linear' -- le alpha varie linéairement de 1.0 à 0.0
                              en horizontal et en vertical à partir du
                              centre jusqu'au bord de l'image
                   None -- le alpha des pixels est égale à 1.0

    Sorties :
        img_warped (np.float32) -- l'image transformée de dimensions = output_shape.
                                   Les valeurs des pixels doivent être dans la
                                   plage [0..1] pour pouvoir visualiser les
                                   résultats avec plt.show(...)

        mask -- tableau numpy de booléens (même dimension que img_warped) indiquant 
                les pixels valides dans l'image de sortie "img_warped"
    """

    image_warped = None
    mask = None
    
    #TODO 3 et 5 : Dans un premier temps (TODO 3), implémentez ici la méthode 
    # qui déforme une image img en applicant dessus la matrice de transformation H. 
    # Vous devez utiliser la projection inverse pour votre implémentation.
    # Pour cela, commencez d'abord par translater les coordonnées de l'image 
    # destination  avec "offset" avant d'appliquer la transformation
    # inverse pour retrouver vos coordonnées dans l'image source.

    # TODO 5 : Dans un deuxième temps, implémentez la partie du code dans cette
    # fonction (contrôlée avec le paramètre 'method' donné ci-dessus) qui calcule 
    # les coefficients du canal alpha de l'image transformée.
    # TODO-BLOC-DEBUT    
    #Rouibah's
    def bilinear_interpolate(img, u,v):
      
      h, w = img.shape[:2] 
    
      u_ = u
      v_ = v
      
      u_ = np.clip(np.array(u_, dtype=np.float64), 0, w - 1) 
      
      v_ = np.clip(np.array(v_, dtype=np.float64), 0, h - 1)

      
      u0 = np.floor(u_).astype(int)
      v0 = np.floor(v_).astype(int)
      u1 = np.clip(u0 + 1,0,w - 1)
      v1 = np.clip(v0 + 1, 0, h - 1)
      wu = (u_ - u0)[:,np.newaxis]  
      wv = (v_ - v0)[:, np.newaxis]  

     
      pvu0_0 = img[v0,u0]  
      pvu1_0 =img[v0,u1]  
      pvu0_1 = img[v1,u0]  
      pvu1_1 = img[v1,u1]  

      
      
      
      return (
          (1 - wu) *(1 - wv) * pvu0_0 +
          wu *(1 - wv) * pvu1_0 +
          (1 - wu) * wv * pvu0_1 +
          wu * wv * pvu1_1
      )
   
     
    
   
    
    
    try:
      H_inv = np.linalg.inv(H)
      
    except np.linalg.LinAlgError:
        
        H_inv = np.identity(3)
    
    
    width_out, height_out = output_shape

    img = img.astype(np.float64) / 255.0 
    src_height, src_width = img.shape[:2]

    
    x = np.arange(width_out)
    y = np.arange(height_out)
    xx, yy = np.meshgrid(x, y)
   

    tr_x = xx +offset[0]
    tr_y = yy +offset[1]

    
    h = np.stack([tr_x, tr_y, np.ones_like(tr_x)]) #ou bien la fonction pad
   
    
    w = H_inv.dot(h.reshape(3, -1))
    w = w.reshape(3, height_out, width_out)
 
    
    epsilon = 1e-10  
    dn = np.where(np.abs(w[2]) < epsilon, 1, w[2]) 
   
    
    u = w[0] / dn
    v = w[1] / dn
  
  
    valid_mask = (u > 0) & (u <= src_width-1) & (v > 0) & (v <= src_height-1) & np.isfinite(u) & np.isfinite(v)
    
   
    
    
    image_warped = np.zeros((height_out, width_out, 4), dtype=np.float64)
  

    for c in range(3):
      bi = bilinear_interpolate(img, u.reshape(-1), v.reshape(-1))
      image_warped[:, :, c] = bi.reshape((height_out, width_out, 3))[:, :, c]  

    alpha = np.ones((height_out, width_out), dtype=np.float64) * valid_mask.astype(np.float64)

    alpha_official = np.ones((height_out, width_out), dtype=np.float64) * valid_mask #oui !
    
    if method is not None:
        
        x = np.arange(src_width + 1)
        y = np.arange(src_height + 1)
        xx, yy = np.meshgrid(x, y)
        x_max = (src_width +1 )/2.0 
        y_max = (src_height +1)/2.0 
        x_center = (src_width /2.0)
        y_center = (src_height / 2.0 )

        dist_x = np.abs(xx - x_center)/(x_max) 
        
        dist_y = np.abs(yy - y_center)/ (y_max)
        
        alpha_x = 1 - dist_x
        alpha_y = 1 - dist_y
        alpha = (alpha_x * alpha_y)[:-1 , :-1]
    
        bi = bilinear_interpolate(alpha[...,np.newaxis], u.reshape(-1), v.reshape(-1))

        alpha = bi.reshape((height_out, width_out)) 
        

       
        alpha = np.clip(alpha,0.0,1.0)
        print(np.max(alpha))
        
        if method == 'linear':
                
            pass 
        elif method == 'hlinear':
            alpha = alpha_x[:-1 , :-1]
        elif method == 'vlinear':
            alpha = alpha_y[:-1 , :-1]

    
        alpha *= valid_mask.astype(np.float64)
       
        image_warped *= alpha_official[..., np.newaxis]
        image_warped[:, :, 3] = alpha

    else:
       
        image_warped *= alpha_official[..., np.newaxis]
        image_warped[:, :, 3] = valid_mask
      

    
    
    mask = valid_mask.astype(bool)  


    

    image_warped = np.clip(image_warped, 0, 1)    
    #H.R
    # TODO-BLOC-FIN
    
    return image_warped, mask


def naive_fusion(img1_warped, img2_warped):
    """
    fusionne deux images selon la formule :
         merged[i,j] = ( image1[i,j] + image2[i,j] ) / (alpha1[i,j]+ alpha2[i,j])
         
    Entrées :
        img1_warped -- Première image RGBA de dimension (Largeur, Heuteur, 4). 
        img2_warped -- Deuxième image RGBA de dimension (Largeur, Heuteur, 4). 
        
    Sorties :
        merged -- image panoramique RGB de dimension (Largeur, Heuteur, 3). 
    """
    
    assert(img1_warped.shape[0] == img2_warped.shape[0] and img1_warped.shape[1] == img2_warped.shape[1] ), \
                 'les images doivent avoir les mêmes dimensions'

    assert(img1_warped.shape[2] == 4 and img2_warped.shape[2] == 4 ), \
                 'les images doivent avoir 4 canaux : R, G, B et A'

    merged = None
    
    #TODO 4 : Implémentez ici la méthode naïve de fusion de deux images en un panorama
    # TODO-BLOC-DEBUT    
    #Rouibah's
    
    def zero_division(d):
        d[d==0] =1
        return d
    img1_rgb = img1_warped[:, :, :3]  
    img2_rgb = img2_warped[:, :, :3]  

    
    alpha1 = img1_warped[:, :, 3:4]   
    alpha2 = img2_warped[:, :, 3:4]   

    d = alpha1 + alpha2

    
   
    d =zero_division(d)

    fusionne = ((img1_rgb * alpha1) + (img2_rgb * alpha2)) / d
    merged = np.clip(fusionne, 0, 1)  
   
    #H.R
    # TODO-BLOC-FIN

    return merged


def fusion(img1_warped, m1, img2_warped, m2):
    """
    fusionne deux images selon la formule :
         merged[i,j] = ( alpha1[i,j] * image1[i,j] + alpha2[i,j] * image2[i,j] ) / (alpha1[i,j]+ alpha2[i,j])
         
    Entrées :
        img1_warped -- Première image RGBA de dimension (Largeur, Heuteur, 4). 
        m1 -- tableau numpy de booléens de dimension (Largeur, Hauteur) indiquant 
                les pixels valides dans l'image img1_warped.
        img2_warped -- Deuxième image RGBA de dimension (Largeur, Heuteur, 4). 
        m2 -- tableau numpy de booléens de dimension (Largeur, Hauteur) indiquant 
                les pixels valides dans l'image img2_warped.

    Sorties :
        merged -- image panoramique RGB de dimension (Largeur, Heuteur, 3). 
    """
        
    assert(img1_warped.shape[0] == img2_warped.shape[0] and img1_warped.shape[1] == img2_warped.shape[1] ), \
                 'les images doivent avoir les mêmes dimensions'

    assert(img1_warped.shape[2] == 4 and img2_warped.shape[2] == 4 ), \
                 'les images doivent avoir 4 canaux : R, G, B et A'
    
    assert(img1_warped.shape[0] == m1.shape[0] and img1_warped.shape[1] == m1.shape[1] ), \
                 'img1_warped et la carte m1 doivent avoir les mêmes dimensions'

    assert(img2_warped.shape[0] == m2.shape[0] and img2_warped.shape[1] == m2.shape[1] ), \
                 'img2_warped et la carte m2 doivent avoir les mêmes dimensions'
    
    merged = None

    #TODO 6 : Implémentez ici la méthode de pondération pour la fusion de deux images en un panorama
    # TODO-BLOC-DEBUT    
    def zero_division(d):
        d[d==0]=1
        return d
    img1_rgb = img1_warped[:, :, :3]  
    img2_rgb = img2_warped[:, :, :3]  

    carte_m1 = m1[:, :, np.newaxis]  #[:, :, None]
    carte_m2 = m2[:, :, np.newaxis] 

    img1_alpha = img1_warped[:, :, 3:4]
    img2_alpha = img2_warped[:, :, 3:4]

    alpha1 = img1_alpha * carte_m1  
    alpha2 = img2_alpha * carte_m2 

    d = alpha1 + alpha2

    
    d=zero_division(d)

    fusionne = ((alpha1 * img1_rgb) + (alpha2 * img2_rgb)) / d
    merged=np.clip(fusionne, 0, 1)     
    
    # H.R    
    # TODO-BLOC-FIN

    return merged


def stitch_multiple_images(imgs_list, keypoints_list, matches_list, imgref=0, blend=None):
    """
    Assemble une liste ordonnée d'images.

    Entrées :
        imgs_list -- Liste d'images à assembler
        keypoints_list -- Liste des tableaux de points-clés. Chaque tableau de points-clés
                est une matrice Mi x 2 de points-clés (x_k,y_k) dans imgs_list[i]. (0 <= k < Mi)
        matches_list -- Liste des tableaux de correspondances. Chaque tableau de correspondances 
                est une matrice N x 2, où chaque rangée représente une correspondance
                [indice_dans_keypoints1, indice_dans_keypoints2] entre les images adjacentes 
                i et i+1 dans imgs_list.
        imgref  -- indice de l'image de référence dans imgs_list.
        blend -- paramètre de sélection de la méthode de calcul des coefficients alpha.
                  'hlinear' -- le alpha varie linéairement de 1.0 à 0.0
                              en horizontal à partir du centre jusqu'au
                              bord de l'image
                  'vlinear' -- le alpha varie linéairement de 1.0 à 0.0
                              en vertical à partir du centre jusqu'au
                              bord de l'image
                  'linear' -- le alpha varie linéairement de 1.0 à 0.0
                              en horizontal et en vertical à partir du
                              centre jusqu'au bord de l'image
                   None -- le alpha des pixels est égale à 1.0

    Sorties :
        panorama : Image panoramique finale. 
    """

    assert ( len(imgs_list) > 1 ), \
        'Nombre d\'images à assembler >= 2'
    
    assert ( len(matches_list) == len(imgs_list) - 1 ), \
        'Nombre des tableaux de correspondances doit être égale à len(imgs_list) - 1'

    assert ( 0 <= imgref and imgref < len(imgs_list) ), \
        'L\'indice de l\'image référence doit être inférieur à len(imgs_list)' 


    panorama = None
    
    #TODO BONUS : Votre implémentation ici
    # TODO-BLOC-DEBUT    
    
    nbr_images = len(imgs_list)
    
    if(nbr_images>=2): 
        Homographys = []
        for i in range(nbr_images - 1):
            H, RMatches = ransac(keypoints_list[i], keypoints_list[i + 1], 
                        matches_list[i], n_iters=500, threshold=1)
            Homographys.append(H)
        
        Homographys = Homographys[::-1]
        transforms = [None] * nbr_images
        transforms[imgref] = np.eye(3) 
        
        #right 
        for i in range(imgref, nbr_images - 1):
            H = Homographys[i]

            if i == imgref:
                transforms[i + 1] = H
            else:
                transforms[i + 1] = transforms[i] @ H
        
        # left 
        for i in range(imgref - 1, -1, -1):

            H = Homographys[i]
            H_inv = np.linalg.inv(H)
            transforms[i] = transforms[i + 1] @ H_inv
       
        output_shape, offset = get_output_space(imgs_list, transforms)
        
        # projection
        warped_images = []
        masks = []

        transforms = transforms[::-1]
        
        for img, H in zip(imgs_list, transforms):
            img_warped, mask = warp_image(img, H, output_shape, offset, method=blend)
            warped_images.append(img_warped)
            masks.append(mask)
        
        
        # Blend 
        panorama = warped_images[0]  
        
        current_mask = masks[0]
       
        for i in range(1, nbr_images):
            
            
            if panorama.shape[2] == 3:  
                alpha_channel = current_mask.astype(np.float64)[..., np.newaxis]
                panorama = np.concatenate([panorama, alpha_channel], axis=2)
            
            
            fusionne = fusion(panorama, current_mask, warped_images[i], masks[i])
          
        
            alpha1 = panorama[:, :, 3:4] * current_mask[..., np.newaxis]
            alpha2 = warped_images[i][:, :, 3:4] * masks[i][..., np.newaxis]
            new_alpha = np.maximum(alpha1, alpha2)
            
            
            panorama = np.concatenate([fusionne, new_alpha], axis=2)
            panorama = np.clip(panorama, 0, 1)
            
            current_mask = current_mask | masks[i]
            
        # RGB 
        panorama = panorama[:, :, :3]  
        
    
    #version : 14-04-2025 at 7:13 PM !

    #Rouibah Hanine
    #HW3 : Reconstruction 3D => Coming Soon !
    #H.R
    # TODO-BLOC-FIN
    
    return panorama
