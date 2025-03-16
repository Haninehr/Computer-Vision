import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

'''
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !!! NE MODIFIEZ PAS LE CODE EN DEHORS DES BLOCS TODO. !!!
 !!!  L'EVALUATEUR AUTOMATIQUE SERA TRES MECHANT AVEC  !!!
 !!!            VOUS SI VOUS LE FAITES !               !!!
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

#def inbounds(shape, indices):
#    assert len(shape) == len(indices)
#    for i, ind in enumerate(indices):
#        if ind < 0 or ind >= shape[i]:
#            return False
#    return True

## Tâche 1 : Détecteur de points-clés ##############################################

class KeypointDetector(object):
    # Implémentez dans les classes enfants
    def detectKeypoints(self, image):
        '''
        Entrée :
            image -- Image uint8 BGR avec des valeurs comprises entre [0, 255]
        Sortie :
            liste des points-clés détectés, remplissez les objets cv2.KeyPoint avec 
            les coordonnées des points-clés détectés, l'angle du gradient (en degrés), 
            la réponse du détecteur (score Harris pour le détecteur Harris) et 
            définissez le paramètre de voisinage 'size' sur 10.
        '''
        raise NotImplementedError()

    # Applique la méthode de suppression non-maximale adaptative
    # pour la sélection des points-clés 
    def selectKeypointsANMS(self, keypoints, maxNbrPoints = 1000):
        '''
        Entrée :
            keypoints -- liste d'objets cv2.KeyPoint des points-clés détectés. 
            nbrPoints -- nombre maximal de points-clés à retouner
        Sortie :
            features -- liste d'objets cv2.KeyPoint des points-clés détectés 
            utilisant la méthode Adaptive Non-Maximal Suppression (ANMS).
        '''
        features = []

        # TODO Bonus : Implémentez ici la méthode de suppression non-maximale 
        # adaptative pour la sélection des points-clés les plus pertinents
        # TODO-BLOC-DEBUT
        #partie Bonus !! not implemented !
        # TODO-BLOC-FIN

        return features

class DummyKeypointDetector(KeypointDetector):
    '''
    Calcul caduc de primitives. Cela ne fait rien de significatif, mais peut 
    être utile à utiliser comme exemple.
    '''
    def detectKeypoints(self, image):
        '''
        Entrée :
            image -- Image uint8 BGR avec des valeurs comprises entre [0, 255]
        Sortie :
            liste des points-clés détectés, remplissez les objets cv2.KeyPoint avec 
            les coordonnées des points-clés détectés, l'angle du gradient (en degrés), 
            la réponse du détecteur (score Harris pour le détecteur Harris) et 
            définissez le paramètre de voisinage 'size' sur 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]
           
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
                 
        #vector = np.vectorize(np.int_)
        #row, col = np.where( vector(255 * (r + g + b) + 0.5) % 100 == 1)
        row, col = np.where(       (255 * (r + g + b) + 0.5).astype(int) % 100 == 1)


        for i in range(np.size(row)):
          (y, x) = (int(row[i]), int(col[i]))

          f = cv2.KeyPoint()
          f.pt = (x, y)
          # Dummy size
          f.size = 10
          f.angle = 0
          f.response = 10

          features.append(f)

        #for y in range(height):
        #    for x in range(width):
        #        r = image[y, x, 0]
        #        g = image[y, x, 1]
        #        b = image[y, x, 2]

        #        if int(255 * (r + g + b) + 0.5) % 100 == 1:
        #            # Si le pixel satisfait ce critère dénué de sens,
        #            # en faire un point-clé.

        #            f = cv2.KeyPoint()
        #            f.pt = (x, y)
        #            # Dummy size
        #            f.size = 10
        #            f.angle = 0
        #            f.response = 10
        #            features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    # Calcule les scores Harris d'une image.
    def computeHarrisValues(self, srcImage):
        '''
        Entrée :
            srcImage -- Image d'entrée en niveaux de gris dans un tableau 
                        numpy avec des valeurs dans [0, 1]. Les dimensions 
                        sont (lignes, cols).

        Sortie :
            harrisImage -- tableau numpy contenant le score de Harris à
                           chaque pixel.

            orientationImage -- tableau numpy contenant l'orientation du gradient 
                                à chaque pixel en degrés.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1 : Calculez l'intensité du coin Harris pour 'srcImage' à
        # chaque pixel et stockez-la dans 'harrisImage'. Calculez également
        # une orientation pour chaque pixel et stockez-la dans 'orientationImage'.
        # TODO-BLOC-DEBUT
        # N'oubliez pas d'enlever ou de commenter la ligne en dessous
        # quand vous implémentez le code de ce TODO
        #Rouibah's
        
        kernel=5
        sigma=0.5
        k=0.1
        demiW = (kernel - 1) // 2
        demiH = (kernel - 1) // 2

        x, y = np.meshgrid(np.arange(kernel) - demiW, np.arange(kernel) - demiH)

       
        weights = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

   
        weights /= weights.sum()

        

        sx = np.array([[ 1 , 2 , 1],
        [ 0,  0,  0],
        [-1 ,-2 ,-1],])
        Ix = ndimage.convolve(srcImage,sx,mode='reflect')

       
        sy = np.array([[ 1 , 0, -1],
        [ 2 , 0 ,-2],
        [ 1  ,0 ,-1]])


        
        Iy = ndimage.convolve(srcImage,sy,mode='reflect')


        
        orientationImage = np.arctan2(Iy, Ix) * 360 / (2*np.pi)
        
        
        Ix2 = Ix * Ix *1
        wIx2 = ndimage.convolve(Ix2, weights, mode='reflect')


        Iy2 = Iy * Iy *1
        wIy2 = ndimage.convolve(Iy2, weights, mode='reflect')
        

        Ixy = Ix * Iy *1
        wIxy = ndimage.convolve(Ixy, weights, mode='reflect')

        # Harris values !
        det = (wIx2 * wIy2) - (wIxy ** 2)
        trace = wIx2 + wIy2
        harrisImage = det - k * (trace ** 2)

       #---------------------------------------- end -------------------------------------------- 
        #H.R                                
        # TODO-BLOC-FIN

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Entrée :
            harrisImage -- tableau numpy contenant le score de Harris à
                           chaque pixel.

        Sortie :
            destImage -- tableau numpy contenant True/False à 
                         chaque pixel, indiquant si la valeur 
                         de celui-ci est un maximum local dans 
                         son voisinage 7x7.
        '''
        destImage = np.zeros_like(harrisImage, bool)

        # TODO 2: Calcul de l'image des maxima locaux
        # TODO-BLOC-DEBUT
        # N'oubliez pas d'enlever ou de commenter la ligne en dessous
        # quand vous implémentez le code de ce TODO
       
        taille = 7
        seuil = 0.001
        local_max = ndimage.maximum_filter(harrisImage, size=taille) #w=7x7
        
        destImage = (harrisImage == local_max) & (harrisImage >= seuil)  

        #H.R              
        # TODO-BLOC-FIN

        return destImage

    def detectKeypoints(self, image):
        '''
        Entrée :
            image -- Image uint8 BGR avec des valeurs comprises entre [0, 255]
        Sortie :
            liste des points-clés détectés, remplissez les objets cv2.KeyPoint avec 
            les coordonnées des points-clés détectés, l'angle du gradient (en degrés), 
            la réponse du détecteur (score Harris pour le détecteur Harris) et 
            définissez le paramètre de voisinage 'size' sur 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Créer une image en niveaux de gris utilisée pour la détection de Harris
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() calcule le score de Harris à chaque position de 
        # pixel, stockant le résultat dans harrisImage.
        # Vous devrez implémenter cette fonction.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Calcule les maxima locaux dans l'image Harris. Vous devrez implémenter 
        # cette fonction. Crée une image pour étiqueter les valeurs maximales 
        # locales de Harris comme Vrai, les autres pixels sur Faux
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Parcourez les points-clés dans harrisMaxImage et remplissez les 
        # informations nécessaires au calcul du descripteur pour chaque point. 
        # Vous devez remplir x, y et angle.

        row, col = np.where(harrisMaxImage == True) 
        
        for i in range(np.size(row)):
          y = int(row[i])
          x = int(col[i])

          f = cv2.KeyPoint()

          # TODO 3 : Remplissez la primitive f avec les données de 
          # position et d'orientation. Initialisez f.size à 10, 
          # f.pt à la coordonnée (x, y), f.angle à l'orientation 
          # en degrés et f.response au score de Harris
          # TODO-BLOC-DEBUT
          # N'oubliez pas d'enlever ou de commenter la ligne en dessous
          # quand vous implémentez le code de ce TODO
          #H.R
          f.size=10
          f.pt=(x,y)
          f.angle=orientationImage[y,x]
          f.response=harrisImage[y,x]
          #---------------------------------------------------
          # TODO-BLOC-FIN

          features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Entrée :
            image -- Image uint8 BGR avec des valeurs comprises entre [0, 255]
        Sortie :
            liste des points-clés détectés, remplissez les objets cv2.KeyPoint avec 
            les coordonnées des points-clés détectés, l'angle du gradient (en degrés), 
            la réponse du détecteur (score Harris pour le détecteur Harris) et 
            définissez le paramètre de voisinage 'size' sur 10.
        '''
        detector = cv2.ORB_create()
        # import pdb; pdb.set_trace()
        return detector.detect(image)


## Tâche 2 : descripteurs de primitives ############################################

class FeatureDescriptor(object):
    # Implémentez dans les classes enfants
    def describeFeatures(self, image, keypoints):
        '''
        Entrée :
            image -- image BGR avec des valeurs comprises entre [0, 255]
            keypoints -- les points-clés détectés, nous devons calculer 
            les descripteurs de primitives aux coordonnées spécifiées 
        Sortie :
            Tableau numpy de descripteurs, dimensions :
                nombre de points-clés x dimension du descripteur
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implémentez des parties de cette fonction
    def describeFeatures(self, image, keypoints):
        '''
        Entrée :
            image -- image BGR avec des valeurs comprises entre [0, 255]
            keypoints -- les points-clés détectés, nous devons calculer 
            les descripteurs de primitives aux coordonnées spécifiées 
        Sortie :
            desc -- Tableau numpy K x 25, où K est le nombre de points-clés,
                    25 est la taille du descripteur
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        for i, f in enumerate(keypoints):
            x, y = int(f.pt[0]), int(f.pt[1])

            # TODO 4 : Le descripteur simple est une fenêtre 5x5 d'intensités 
            # centrée sur le point d'intérêt. Stockez le descripteur en 
            # tant que vecteur-ligne dans le tableau numpy. Traitez les 
            # pixels à l'extérieur de l'image comme des zéros.
            # TODO-BLOC-DEBUT            
            # N'oubliez pas d'enlever ou de commenter la ligne en dessous
            # quand vous implémentez le code de ce TODO
           
            Pa = np.pad(grayImage, [(2, 2), (2, 2)], mode='constant')
            desc_ = Pa[y:y+5, x:x+5]
            desc_=desc_.reshape(1,25)
        
            #-------------------------------------------------------
            #H.R
            # TODO-BLOC-FIN

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implémentez des parties de cette fonction
    def describeFeatures(self, image, keypoints):
        '''
        Entrée :
            image -- image BGR avec des valeurs comprises entre [0, 255]
            keypoints -- les points-clés détectés, nous devons calculer 
            les descripteurs de primitives aux coordonnées spécifiées 
        Sortie :
            desc -- Tableau numpy K x W^2, où K est le nombre de points-clés
                    et W est la taille de la fenêtre
        '''

        image = image.astype(np.float32)
        image /= 255.
        # Cette image représente la fenêtre autour du point-clé que 
        # vous devez utiliser pour calculer le descripteur de primitive 
        # (stockée ligne par ligne)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            #TODO 5 : Calculez la transformation selon l'emplacement et 
            # l'orientation du point-clé. Vous devez calculer la transformation 
            # pour chaque pixel de la fenêtre 40x40 pivotée et entourant 
            # le point-clé vers les pixels appropriés dans l'image du 
            # descripteur de primitive 8x8.
            transMx = np.zeros((2, 3))

            # TODO-BLOC-DEBUT
            # N'oubliez pas d'enlever ou de commenter la ligne en dessous
            # quand vous implémentez le code de ce TODO
            #H.R
            x,y = f.pt
            
            angle = np.radians(-f.angle) #simple !
            
            #translation vers le center du point clé !
            tr_mx1 = np.array([[1,0,-float(x)],[0,1,-float(y)],[0,0,1]])


          
            rot_mx = np.eye(3) 
            rot_mx[:2, :2] = [[np.cos(angle), -np.sin(angle)], 
                            [np.sin(angle),  np.cos(angle)]]

            scale_mx = np.array([[windowSize/40.0,0,0],
                                 [0,windowSize/40.0,0],
                                 [0,0,1]])

            tr_mx2 = np.array([[1,0,windowSize//2], [0,1,windowSize//2], [0,0,1]])


            transMx=(tr_mx2 @ scale_mx @ rot_mx @ tr_mx1)[:2,:] #2x3 ! as needed !!
            #T2 S R T1
            #H.R
            # TODO-BLOC-FIN

            # Appel la fonction de distorsion affine pour effectuer le mappage
            # elle requiert une matrice 2x3
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_AREA)

            # TODO 6 : Normalisez le descripteur pour avoir une moyenne nulle
            # et une variance égale à 1. Si la variance avant normalisation 
            # est négligeable (que nous définissons comme inférieure à 1e-10), 
            # alors affectez zéro au descripteur. Enfin, stockez le descripteur 
            # dans le tableau 'desc'.
            # TODO-BLOC-DEBUT
            # N'oubliez pas d'enlever ou de commenter la ligne en dessous
            # quand vous implémentez le code de ce TODO
            std = np.std(destImage)  
            
            if np.abs(std ** 2) < 1e-10:  
                desc[i] = np.zeros_like(destImage).flatten()  # null
               
            else:
                desc[i] = ((destImage - np.mean(destImage)) / std).flatten()  # Normalisation !

            #H.R
            # TODO-BLOC-FIN

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Entrée :
            image -- image BGR avec des valeurs comprises entre [0, 255]
            keypoints -- les points-clés détectés, nous devons calculer 
            les descripteurs de primitives aux coordonnées spécifiées 
        Sortie :
            Tableau numpy de descripteurs, dimensions :
                nombre de points-clés x dimension du descripteur
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


## Tâche 3 : mise en correspondance de primitives ##################################

class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Entrée :
            desc1 -- les descripteurs de primitives de l'image 1 stockés dans un tableau numpy,
                dimensions:   lignes   (nombre de points-clés)
                            x colonnes (dimension du descripteur)
            desc2 -- les descripteurs de primitives de l'image 1 stockés dans un tableau numpy,
                dimensions:   lignes   (nombre de points-clés)
                            x colonnes (dimension du descripteur)
        Sortie :
            liste des correspondances : une liste d'objets cv2.DMatch
                Comment définir les attributs :
                    queryIdx : L'index de la primitive dans la première image
                    trainIdx : L'index de la primitive dans la seconde image
                    distance : La distance entre les deux primitives
        '''
        raise NotImplementedError

    # Évalue une paire de correspondance en se basant sur une homographie 
    # connue. Cette fonction calcule la distance euclidienne moyenne entre les 
    # points-clés appariés et les positions réelles transformées par 
    # homographie.

    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Distance euclidienne 
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transformation d'un point par homographie.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Entrée :
            desc1 -- les descripteurs de primitives de l'image 1 stockés dans un tableau numpy,
                dimensions:   lignes   (nombre de points-clés)
                            x colonnes (dimension du descripteur)
            desc2 -- les descripteurs de primitives de l'image 2 stockés dans un tableau numpy,
                dimensions:   lignes   (nombre de points-clés)
                            x colonnes (dimension du descripteur)
        Sortie :
            liste des correspondances : une liste d'objets cv2.DMatch
                Comment définir les attributs :
                    queryIdx : L'index de la primitive dans la première image
                    trainIdx : L'index de la primitive dans la seconde image
                    distance : La distance SMC entre les deux primitives
        '''
        matches = []
        # nombre de primitive = n
        assert desc1.ndim == 2
        # nombre de primitive = m
        assert desc2.ndim == 2
        # les deux descripteurs doivent avoir les mêmes dimensions
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7 : Effectuez une mise en correspondance simple des primitives. 
        # Faites correspondre une primitive de la première image avec la primitive 
        # la plus proche de la seconde image on utilisant la distance SMC entre
        # descripteurs.
        # TODO-BLOC-DEBUT
        # N'oubliez pas d'enlever ou de commenter la ligne en dessous
        # quand vous implémentez le code de ce TODO

        dist = spatial.distance.cdist(desc1, desc2, metric='sqeuclidean')


        length = desc1.shape[0]
        for i in range(length):
            match = np.argmin(dist[i])  

         
            bidiric = np.argmin(dist[:, match])
            if bidiric != i:
               
                continue  

            matches.append(cv2.DMatch(i, int(match), float(dist[i, match])))
        #H.R
        # TODO-BLOC-FIN

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Entrée :
            desc1 -- les descripteurs de primitives de l'image 1 stockés dans un tableau numpy,
                dimensions:   lignes   (nombre de points-clés)
                            x colonnes (dimension du descripteur)
            desc2 -- les descripteurs de primitives de l'image 2 stockés dans un tableau numpy,
                dimensions:   lignes   (nombre de points-clés)
                            x colonnes (dimension du descripteur)
        Sortie :
            liste des correspondances : une liste d'objets cv2.DMatch
                Comment définir les attributs :
                    queryIdx : L'index de la primitive dans la première image
                    trainIdx : L'index de la primitive dans la seconde image
                    distance : Le rapport de distance entre les deux primitives
        '''

        matches = []
        # nombre de primitive = n
        assert desc1.ndim == 2
        # nombre de primitive = m
        assert desc2.ndim == 2
        # les deux descripteurs doivent avoir les mêmes dimensions
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8 : Effectuez une mise en correspondance des primitives 
        # utilisant le rapport de distance entre primitives.
        # Faites correspondre une primitive de la première image avec la primitive 
        # la plus proche de la seconde image on utilisant le rapport de distance 
        # entre les deux meilleurs descripteurs correspondants.        
        # Utilisez un seuil de 0.7 pour sélectionner les 'bonnes' correspondances
        # TODO-BLOC-DEBUT
        # N'oubliez pas d'enlever ou de commenter la ligne en dessous
        # quand vous implémentez le code de ce TODO
        
        #Note : this is garbage !Not Working !
        #try to Solove it ! 
        
        distance = spatial.distance.cdist(desc1, desc2, 'euclidean') 
       
        cosine_simularit = 1 - spatial.distance.cdist(desc1, desc2, 'cosine')  # Similarité cosinus, donc entre 0 et 1 (au lieu de -1 et 1 !)
        
        best_distances = []
        #forward_matches = {}

        for queryIdx,row in enumerate(distance):
            
            if len(row) < 2:  # au moins 2 correspondances
              
                continue #éliminer !

            rowIdx = np.argpartition(row, 1)[:2] 
            premiere, seconde  = rowIdx[np.argsort(row[rowIdx])] 
            if row[seconde] == 0 or row[seconde] is None or np.isnan(row[seconde]) :  #avoid !division par zéro
                
                continue #f

            
            if np.isclose(row[premiere],row[seconde]):
                #éliminer ce point ! aussi !
                #trés proche !
                continue

            if cosine_simularit[queryIdx, premiere] < 0.9:
                #c est precque inutile !
                continue

            
            

            ratio = row[premiere] / float(row[seconde]) #lowe C
            
            

           
            if (ratio < 0.7 ):  
                
                 
                    matches.append(cv2.DMatch(queryIdx, premiere, float(ratio)))
                    best_distances.append(row[premiere])
                    
            
            if len(matches) > 3: 

                
                τ = np.median(best_distances)
                
                #filtre
                matches = [m for m in matches if m.distance <= τ]
                
        
        #Rouibah Hanine !
        #Project 2 : Image Panoramiques => Coming soon !
       
        # TODO-BLOC-END

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))
