import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

from stereo import epipolar_match


def undistort_image(img, principal_point, focal_distance, distCoeffs):
    """reduce distortions in projected img.
    Parameters
    ----------
    img : (N, M [, 3]) array
        grayscale or color image.
    principal_point : (1,2) tuple
        (cx,cy) coordinates of the camera principal point
    focal_distance : (1,2) tuple
        (fx,fy) focal lengths of the camera             
    distCoeffs : (1, L) array
        OpenCV distortion coefficients ovtained from camera Calibration
    """
    
    mtx = np.eye(3)
    mtx[:2,2] = principal_point
    np.fill_diagonal(mtx[:2,:2], focal_distance)
    #src : https://github.com/YvanYin/DrivingWorld/blob/faceb9861582c2ecad0d82c1f97699a1a314cb62/utils/undistort_image.py#L38
    #src : https://github.com/JiahaoPlus/camera-calibration/blob/master/point2d_3d.py
    #camera intrinsic matrix
    #[[focal_distance[0]   0.  pricipal_point[0]]  
    #[  0. focal_distance[1]   princpal_point[1]]
    #[  0.   0.   1.]]

    #[[fx   0. cx]
    #[  0.  fy  cy]
    #[  0.   0.   1.]]
        
    # undistort
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distCoeffs, (w,h), 1, (w,h))
    img = cv2.undistort(img, mtx, distCoeffs, None, new_mtx)
        
    # crop the image
    x, y, w, h = roi
    img = img[y:y+h, x:x+w]

    return img


def plot_matches(ax, image1, image2, keypoints1, keypoints2, matches,
                 keypoints_color='k', matches_color=None, only_matches=False):
    """Plot matched features.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matches and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    keypoints1 : (K1) List
        OpenCV keypoint objects in image1 as a List.
    keypoints2 : (K2) List
        OpenCV keypoint objects in image2 as a List.
    matches : (Q, 2) array
        indices of corresponding matches in first and second set of
        keypoints, where ``keypoints1[ matches|:, 0] , :]`` denotes the 
        coordinates of the first and ``keypoints2[ matches|:, 1] , :]`` the 
        coordinates of the second set of keypoints.
        
    keypoints_color : matplotlib color, optional
        Color for keypoint locations.
    matches_color : matplotlib color, optional
        Color for lines which connect keypoint matches. By default the
        color is chosen randomly.
    only_matches : bool, optional
        Whether to only plot matches and not plot the keypoint locations.
    """

    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    image = np.concatenate([image1, image2], axis=1)

    offset = image1.shape

    ax.imshow(image, interpolation='nearest', cmap='gray')
    ax.axis((0, 2 * offset[1], offset[0], 0))
    
    if not only_matches:
        pts1 = np.squeeze( np.array([key_point.pt for key_point in keypoints1]).reshape(-1, 1, 2) )
        ax.scatter(pts1[:, 0], pts1[:, 1],
                   facecolors='none', edgecolors=keypoints_color)

        pts2 = np.squeeze( np.array([key_point.pt for key_point in keypoints2]).reshape(-1, 1, 2) )
        ax.scatter(pts2[:, 0] + offset[1], pts2[:, 1],
                   facecolors='none', edgecolors=keypoints_color)                

    for m in matches:
        if matches_color is None:
            color = np.random.rand(3)
        else:
            color = matches_color
                
        (x1, y1) = keypoints1[m[0]].pt
        (x2, y2) = keypoints2[m[1]].pt

        ax.plot([x1, x2 + offset[1]], [y1, y2],'-', color=color)             

        
class DisplayEpipolarLine:
    def __init__(self, img_l, img_r, F, showMatch=False, showEpiLines=False):
        """Display Epipolar lines
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matches and image are drawn in this ax.
        img_l : (N, M [, 3]) array
            First grayscale or color image.
        img_r : (N, M [, 3]) array
            Second grayscale or color image.
        F : (3, 3) array
            Fundamental matrix.
        
        showMatch : bool, optional
            show matching point in second image.
        showRpiLines : bool, optional
            show epipolar lines for randomly selected points.
        """

        self.showMatch       = showMatch  
        self.F               = F
        self.Img_l           = img_l
        self.Img_r           = img_r
        (self.sy , self.sx ) = img_r.shape[:2]
       
        self.fig, [self.ax_l, self.ax_r] = plt.subplots(1, 2)
        
        # Hide the Figure name at the top of the figure
        self.fig.canvas.header_visible = False
        # Hide the footer
        self.fig.canvas.footer_visible = False
                                
        self.ax_l.imshow( cv2.cvtColor(img_l, cv2.COLOR_RGB2GRAY), cmap='gray')
        self.ax_l.set_axis_off()
        
        self.ax_r.imshow(cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY), cmap='gray')
        self.ax_r.set_axis_off()        
                
        if showEpiLines == True:
            h, w = img_l.shape[:2]

            #choix aléatoire de conq points à visualiser
            x_coords = np.random.randint(w//2 - w//3, w//2 + w//3, size=5)
            y_coords = np.random.randint(w//2 - w//3, h//2 + w//3, size=5)

            self.ax_l.set_title('des points sélectionnés\naléatoirement dans cette image', y=-0.15)
            self.ax_r.set_title('lignes épipolaires associées (même couleur) \n sur cette image', y=-0.15)            

            points = np.vstack( (x_coords, y_coords, np.ones(len(x_coords)) ))

            for i in range(points.shape[1]):
                self.draw_line(points[:,i])
            self.fig.canvas.draw()

        else:
            if not showMatch:
                self.ax_r.set_title('Vérifiez que le point correspondant \n est sur la ligne épipolaire dans cette image', y=-0.15)
            else:
                self.ax_r.set_title('Vérifiez que le point correspondant \n est là où il faut qu\'il soit sur cette image', y=-0.15)            

            self.ax_l.set_title('Sélectionnez un point dans cette image', y=-0.10)
            self.cid_press = self.ax_l.get_figure().canvas.mpl_connect('button_press_event', self)

        plt.tight_layout()
        plt.show()

    def __call__(self, event):                
        """Mouse event handler
        Parameters
        ----------
        event : matplotlib. buttom press event
            contains the x,y coordinates of the selected (clicked) pixel in img_l
        """

        if event.inaxes!=self.ax_l: return

        v = np.array([[event.xdata], [event.ydata], [1.]])
        self.draw_line(v)
        self.fig.canvas.draw()

    def draw_line(self, v):
        """Draw line function
        Parameters
        ----------
        v : (3, 1) array 
            homogeneous coordinates of the selected point in img_l
        """

        l = ( self.F @ v ).flatten()
                
        s = np.sqrt(l[0]**2+l[1]**2)        
        if s==0:
            print('Zero line vector in DisplayEpipolarLine')
            return
              
        #definir les coordonnées des quatre coins de l'image
        tl = np.array([      0.0,       0.0, 1.0])
        tr = np.array([self.sx-1,       0.0, 1.0])
        br = np.array([self.sx-1, self.sy-1, 1.0])
        bl = np.array([      0.0, self.sy-1, 1.0])
                    
        #calcul des equations des quatre lignes du cadre de l'image 
        top    = np.cross( tr, tl )
        bottom = np.cross( br, bl )
        left   = np.cross( tl, bl )
        right  = np.cross( tr, br )
                
        #points d'intersection de la ligne epipolaire avec les lignes du cadre
        pt         = np.zeros((4,3))
        pt[0, :]   = np.cross(l, top)
        pt[1, :]   = np.cross(l, bottom)        
        pt[2, :]   = np.cross(l, left)
        pt[3, :]   = np.cross(l, right)

        #
        # determiner les coordonnées de début (xs,ys) et de fin (xe,ye) de la ligne epipolaire
        #
        if math.isclose(pt[0, 2], 0.0) or math.isclose(pt[1, 2], 0.0):
            #pt is at horizontal infinity (+ or -) -> l is parallel with horizontal axis             
            pt[2, :2] /= pt[2, 2]
            pt[3, :2] /= pt[3, 2]

            idx = np.argsort(pt[2:, 0])
            (xs, ys) = pt[2+idx[0], :2]
            (xe, ye) = pt[2+idx[1], :2]
                        
        elif math.isclose(pt[2, 2], 0.0) or math.isclose(pt[3, 2], 0.0):
            #pt is at vertical infinity (+ or -) -> l is parallel with vertical axis                         
            pt[0, :2] /= pt[0, 2]
            pt[1, :2] /= pt[1, 2]
            
            idx = np.argsort(pt[:2, 0])
            (xs, ys) = pt[idx[0], :2]
            (xe, ye) = pt[idx[1], :2]
            
        else:
            # general case (l is oblique )
            pt[0, :2] /= pt[0, 2]
            pt[1, :2] /= pt[1, 2]
            pt[2, :2] /= pt[2, 2]
            pt[3, :2] /= pt[3, 2]
                        
            idx = np.argsort(pt[:, 0])
        
            (xs, ys) = pt[idx[1], :2]
            (xe, ye) = pt[idx[2], :2]
        
        #
        # visualisation du point sélectionné et de la ligne epipolaire correspondante
        #
        self.ax_l.plot(v[0], v[1], '*', markersize=6, linewidth=2)        
        self.ax_r.plot([xs, xe], [ys, ye], linewidth=2)                
         
        # visualisation du point correspondant dans la ligne epipolaire
        if True == self.showMatch:
            # draw matching point
            pc = v[:2].T    
            pm = epipolar_match(self.Img_l, self.Img_r, self.F, pc)
            self.ax_r.plot(pm[0,0], pm[0,1], 'ro', markersize=8, linewidth=2)
        
                
def _projtrans(H, p):
    n = p.shape[1]
    p3d = np.vstack((p, np.ones((1,n))))
    h2d = H @ p3d
    p2d = h2d[:2,:] / np.vstack((h2d[2,:], h2d[2,:]))
    return p2d


def _mcbbox(s1, s2, H1, H2):
    c1 = np.array([[0,     0, s1[1], s1[1]], 
                   [0, s1[0],     0, s1[0]]])
    
    c1p = _projtrans(H1, c1)
    
    #minx, miny, maxx, maxy
    bb1 = [np.floor(np.amin(c1p[0,:])),  
           np.floor(np.amin(c1p[1,:])),
           np.ceil(np.amax(c1p[0,:])),
           np.ceil(np.amax(c1p[1,:]))]

    # size of the output image 1
    sz1 = [bb1[2] - bb1[0], 
           bb1[3] - bb1[1]]
    
    #
    #
    c2 = np.array([[0,     0, s2[1], s2[1]], 
                   [0, s2[0],     0, s2[0]]])
    
    c2p = _projtrans(H2, c2)

    #minx, miny, maxx, maxy    
    bb2 = [np.floor(np.amin(c2p[0,:])),
           np.floor(np.amin(c2p[1,:])),
           np.ceil(np.amax(c2p[0,:])),
           np.ceil(np.amax(c2p[1,:]))]
    
    # size of the output image 2
    sz2 = [bb2[2] - bb2[0], 
           bb2[3] - bb2[1]]
        
    sz    = np.vstack((sz1, sz2))
    szmax = np.amax(sz, axis=0)
        
    return szmax, bb1[:2], bb2[:2]


def _warpStereo(I1, I2, H1, H2):
    
    sz, tl1, tl2 = _mcbbox(I1.shape, I2.shape, H1, H2)        
    
    miny = min(tl1[1], tl2[1])         
    T1 = np.array([[1, 0, -tl1[0]], [0, 1, -miny], [0,0,1]])
    T2 = np.array([[1, 0, -tl2[0]], [0, 1, -miny], [0,0,1]])

    sz  = (int(sz[0]), int(sz[1]))
    
    I1p = cv2.warpPerspective(I1, T1 @ H1, sz)
    I2p = cv2.warpPerspective(I2, T2 @ H2, sz)
        
    return I1p, I2p, T1, T2
    
    
class CameraPoseVisualizer:
    def __init__(self, Rt_list, pts3d_list = None, scale=0.2):
        assert len(Rt_list) == 4, 'Une liste de quatre poses R|t est requise' 

        if pts3d_list is not None:
            assert len(pts3d_list) == 4, 'Une liste de quatre reconstructions pts3d est requise' 
            
        self.fig , self.axes = plt.subplots(nrows=2, ncols=2, subplot_kw={'projection':'3d'}) #figsize=(9, 4)
        self.axes = self.axes.ravel()

        # define reference camera
        R1, C1 = np.eye(3), np.zeros((3, 1))
    
        for i, Rt in enumerate(Rt_list):
            R2 = Rt[:,:3]
            t  = Rt[:,3]
            C2 =(-R2.T @ t).reshape(-1,1)
        
            self.draw_camera(self.axes[i], R1, C1, scale, linecol='b-')
            self.draw_camera(self.axes[i], R2, C2, scale)
            
            if pts3d_list is not None:
                X = pts3d_list[i]
                self.axes[i].plot(X[:, 0], X[:, 1], X[:, 2], 'b.')

            self.set_axes_equal(self.axes[i])
            self.axes[i].set_xlabel('x axis')
            self.axes[i].set_ylabel('y axis')
            self.axes[i].set_zlabel('z axis')
            self.axes[i].view_init(azim=-90, elev=0)
        self.fig.tight_layout()
        
        if pts3d_list is None:
            plt.title('') #Camera pose
        else:
            plt.title('') #Camera pose witn projected 3d points                       
        plt.show()
                    
    def set_axes_equal(self, ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
        y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
        z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

        plot_radius = 0.5*max([x_range, y_range, z_range])
  
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        
    def draw_camera(self, ax, R, C, scale, linecol='k-'):
        axis_end_points = C + scale * R.T  # (3, 3)
        vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
        vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

        C = C.ravel()
        # draw coordinate system of camera
        ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
        ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
        ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

        # draw square window and lines connecting it to camera center
        ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], linecol)
        ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], linecol)
        ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], linecol)
        ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], linecol)
        ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], linecol)
                    

            
            
