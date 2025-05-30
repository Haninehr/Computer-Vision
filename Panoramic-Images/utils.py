import numpy as np

# Functions to convert points to homogeneous coordinates and back
pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:,:-1]

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

    image1.astype(np.float32)
    image2.astype(np.float32)

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