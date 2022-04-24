import math
import tensorflow as tf

def random_rotation(image, max_degrees, bbox=None, prob=0.5):
    """Applies random rotation to image and bbox"""
    def _rotation(image, bbox):
        # Get random angle
        degrees = tf.random.uniform([], minval=-max_degrees, maxval=max_degrees, dtype=tf.float32)
        radians = degrees * math.pi / 180.
        if bbox is not None:
            # Get offset from image center
            image_shape = tf.cast(tf.shape(input=image), tf.float32)
            image_height, image_width = image_shape[0], image_shape[1]
            bbox = tf.cast(bbox, tf.float32)
            center_x = image_width / 2.
            center_y = image_height / 2.
            bbox_center_x = (bbox[0] + bbox[2]) / 2.
            bbox_center_y = (bbox[1] + bbox[3]) / 2.
            trans_x = center_x - bbox_center_x
            trans_y = center_y - bbox_center_y

            # Apply rotation
            image = _translate_image(image, trans_x, trans_y)
            bbox = _translate_bbox(bbox, image_height, image_width, trans_x, trans_y)
            image = tf.contrib.image.rotate(image, radians, interpolation='BILINEAR')
            bbox = _rotate_bbox(bbox, image_height, image_width, radians)
            image = _translate_image(image, -trans_x, -trans_y)
            bbox = _translate_bbox(bbox, image_height, image_width, -trans_x, -trans_y)
            bbox = tf.cast(bbox, tf.int32)

            return image, bbox
        return tf.contrib.image.rotate(image, radians, interpolation='BILINEAR')

    retval = image if bbox is None else (image, bbox)
    return tf.cond(pred=_should_apply(prob), true_fn=lambda: _rotation(image, bbox), false_fn=lambda: retval)


def random_bbox_jitter(bbox, image_height, image_width, max_fraction, prob=0.5):
    """Randomly jitters bbox coordinates by +/- jitter_fraction of the width/height"""
    def _bbox_jitter(bbox):
        bbox = tf.cast(bbox, tf.float32)
        width_jitter = max_fraction*(bbox[2] - bbox[0])
        height_jitter = max_fraction*(bbox[3] - bbox[1])
        xmin = bbox[0] + tf.random.uniform([], minval=-width_jitter, maxval=width_jitter, dtype=tf.float32)
        ymin = bbox[1] + tf.random.uniform([], minval=-height_jitter, maxval=height_jitter, dtype=tf.float32)
        xmax = bbox[2] + tf.random.uniform([], minval=-width_jitter, maxval=width_jitter, dtype=tf.float32)
        ymax = bbox[3] + tf.random.uniform([], minval=-height_jitter, maxval=height_jitter, dtype=tf.float32)
        xmin, ymin, xmax, ymax = _clip_bbox(xmin, ymin, xmax, ymax, image_height, image_width)
        bbox = tf.cast(tf.stack([xmin, ymin, xmax, ymax]), tf.int32)
        return bbox

    return tf.cond(pred=_should_apply(prob), true_fn=lambda: _bbox_jitter(bbox), false_fn=lambda: bbox)


def random_shift_and_scale(image, max_shift, max_scale_change, prob=0.5):
    """Applies random shift and scale to pixel values"""
    def _shift_and_scale(image):
        shift = tf.cast(tf.random.uniform([], minval=-max_shift, maxval=max_shift, dtype=tf.int32), tf.float32)
        scale = tf.random.uniform([], minval=(1. - max_scale_change),
                                  maxval=(1. + max_scale_change), dtype=tf.float32)
        image = scale*(tf.cast(image, tf.float32) + shift)
        image = tf.cast(tf.clip_by_value(image, 0., 255.), tf.uint8)
        return image

    return tf.cond(pred=_should_apply(prob), true_fn=lambda: _shift_and_scale(image), false_fn=lambda: image)


def random_shear(image, max_lambda, bbox=None, prob=0.5):
    """Applies random shear in either the x or y direction"""
    shear_lambda = tf.random.uniform([], minval=-max_lambda, maxval=max_lambda, dtype=tf.float32)
    image_shape = tf.cast(tf.shape(input=image), tf.float32)
    image_height, image_width = image_shape[0], image_shape[1]

    def _shear_x(image, bbox):
        image = _shear_x_image(image, shear_lambda)
        if bbox is not None:
            bbox = _shear_bbox(bbox, image_height, image_width, shear_lambda, horizontal=True)
            bbox = tf.cast(bbox, tf.int32)
            return image, bbox
        return image

    def _shear_y(image, bbox):
        image = _shear_y_image(image, shear_lambda)
        if bbox is not None:
            bbox = _shear_bbox(bbox, image_height, image_width, shear_lambda, horizontal=False)
            bbox = tf.cast(bbox, tf.int32)
            return image, bbox
        return image

    def _shear(image, bbox):
        return tf.cond(pred=_should_apply(0.5), true_fn=lambda: _shear_x(image, bbox), false_fn=lambda: _shear_y(image, bbox))

    retval = image if bbox is None else (image, bbox)
    return tf.cond(pred=_should_apply(prob), true_fn=lambda: _shear(image, bbox), false_fn=lambda: retval)

def exterior_exclusion(image):
    """Removes visual features exterior to the patient's body"""
    # Create initial binary image
    filt_image = cv2.GaussianBlur(image, (5, 5), 0)
    filt_image.shape = image.shape  # ensure channel dimension is preserved if present
    thresh = cv2.threshold(filt_image[filt_image > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    bin_image = filt_image > thresh

    # Find body contour
    body_cont = body_contour(bin_image.astype(np.uint8))

    # Exclude external regions by replacing with bg mean
    body_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(body_mask, [body_cont], 0, 1, -1)
    body_mask = body_mask.astype(bool)
    bg_mask = (~body_mask) & (image > 0)
    bg_dark = bg_mask & (~bin_image)  # exclude bright regions from mean
    bg_mean = np.mean(image[bg_dark])
    image[bg_mask] = bg_mean
    return image

def random_exterior_exclusion(image, prob=0.5):
    """Randomly removes visual features exterior to the patient's body"""
    def _exterior_exclusion(image):
        shape = image.get_shape()
        image = tf.numpy_function(exterior_exclusion, [image], tf.uint8)
        image.set_shape(shape)
        return image
    return tf.cond(pred=_should_apply(prob), true_fn=lambda: _exterior_exclusion(image), false_fn=lambda: image)


def _translate_image(image, delta_x, delta_y):
    """Translate an image"""
    return tf.contrib.image.translate(image, [delta_x, delta_y], interpolation='BILINEAR')


def _translate_bbox(bbox, image_height, image_width, delta_x, delta_y):
    """Translate an bbox, ensuring coordinates lie in the image"""
    bbox = bbox + tf.stack([delta_x, delta_y, delta_x, delta_y])
    xmin, ymin, xmax, ymax = _clip_bbox(bbox[0], bbox[1], bbox[2], bbox[3], image_height, image_width)
    bbox = tf.stack([xmin, ymin, xmax, ymax])
    return bbox


def _rotate_bbox(bbox, image_height, image_width, radians):
    """Rotates the bbox by the given angle"""
    # Shift bbox to origin
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    center_x = (xmin + xmax) / 2.
    center_y = (ymin + ymax) / 2.
    xmin = xmin - center_x
    xmax = xmax - center_x
    ymin = ymin - center_y
    ymax = ymax - center_y

    # Rotate bbox coordinates
    radians = -radians  # negate direction since y-axis is flipped
    coords = tf.stack([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]])
    coords = tf.transpose(a=tf.cast(coords, tf.float32))
    rotation_matrix = tf.stack(
        [[tf.cos(radians), -tf.sin(radians)],
         [tf.sin(radians), tf.cos(radians)]])
    new_coords = tf.matmul(rotation_matrix, coords)

    # Find new bbox coordinates and clip to image size
    xmin = tf.reduce_min(input_tensor=new_coords[0, :]) + center_x
    ymin = tf.reduce_min(input_tensor=new_coords[1, :]) + center_y
    xmax = tf.reduce_max(input_tensor=new_coords[0, :]) + center_x
    ymax = tf.reduce_max(input_tensor=new_coords[1, :]) + center_y
    xmin, ymin, xmax, ymax = _clip_bbox(xmin, ymin, xmax, ymax, image_height, image_width)
    bbox = tf.stack([xmin, ymin, xmax, ymax])

    return bbox


def _shear_x_image(image, shear_lambda):
    """Shear image in x-direction"""
    tform = tf.stack([1., shear_lambda, 0., 0., 1., 0., 0., 0.])
    image = tf.contrib.image.transform(
        image, tform, interpolation='BILINEAR')
    return image


def _shear_y_image(image, shear_lambda):
    """Shear image in y-direction"""
    tform = tf.stack([1., 0., 0., shear_lambda, 1., 0., 0., 0.])
    image = tf.contrib.image.transform(
        image, tform, interpolation='BILINEAR')
    return image


def _shear_bbox(bbox, image_height, image_width, shear_lambda, horizontal=True):
    """Shear bbox in x- or y-direction"""
    # Shear bbox coordinates
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    coords = tf.stack([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]])
    coords = tf.transpose(a=tf.cast(coords, tf.float32))
    if horizontal:
        shear_matrix = tf.stack(
            [[1., -shear_lambda],
             [0., 1.]])
    else:
        shear_matrix = tf.stack(
            [[1., 0.],
             [-shear_lambda, 1.]])
    new_coords = tf.matmul(shear_matrix, coords)

    # Find new bbox coordinates and clip to image size
    xmin = tf.reduce_min(input_tensor=new_coords[0, :])
    ymin = tf.reduce_min(input_tensor=new_coords[1, :])
    xmax = tf.reduce_max(input_tensor=new_coords[0, :])
    ymax = tf.reduce_max(input_tensor=new_coords[1, :])
    xmin, ymin, xmax, ymax = _clip_bbox(xmin, ymin, xmax, ymax, image_height, image_width)
    bbox = tf.stack([xmin, ymin, xmax, ymax])

    return bbox


def _clip_bbox(xmin, ymin, xmax, ymax, image_height, image_width):
    """Clip bbox to valid image coordinates"""
    xmin = tf.clip_by_value(xmin, 0, image_width)
    ymin = tf.clip_by_value(ymin, 0, image_height)
    xmax = tf.clip_by_value(xmax, 0, image_width)
    ymax = tf.clip_by_value(ymax, 0, image_height)
    return xmin, ymin, xmax, ymax


def _should_apply(prob):
    """Helper function to create bool tensor with probability"""
    return tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
