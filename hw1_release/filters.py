import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)

    center_X = int(Wk / 2)
    center_Y = int(Hk / 2)

    for X in range(Wi):
        for Y in range(Hi):
            for xk in range(Wk):
                for yk in range(Hk):
                    global_x = X + xk - center_X
                    global_y = Y + yk - center_Y

                    if 0 <= global_x < Wi and 0 <= global_y < Hi:
                        out[Y, X] = out[Y, X] + image[global_y, global_x] * kernel[yk, xk]


    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.pad(image, ((pad_height,),(pad_width,)), mode = 'constant')
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)
    pad_height = int(Hk/2)
    pad_width = int(Wk/2)
    img_pad = zero_pad(image, pad_height, pad_width)
    for h in range(Hi):
        for w in range(Wi):
            # define the current slice
            img_slice = img_pad[h:h + Hk,w:w + Wk]
            out[h,w] = np.sum(img_slice * kernel)

    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    from scipy import signal
    out = signal.convolve2d(image, kernel, mode='same')
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(np.flip(g,0),1)
    out = conv_fast(f,g)

    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    mean = np.mean(g)
    g -=mean
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    Hf,Wf = f.shape
    Hg,Wg = g.shape
    mean_g = np.mean(g)
    std_g = np.std(g)
    norm_kernel = (g - mean_g)/ std_g
    img_pad = zero_pad(f, int(Hg/2), int(Wg/2))
    out = np.zeros(f.shape)
    for h in range(Hf):
        for w in range(Wf):
            img_slice =img_pad[h:h + Hg,w:w + Wg]
            norm_img = (img_slice - np.mean(img_slice)) / np.std(img_slice)
            out[h,w] = np.sum(norm_img * norm_kernel)
    ### END YOUR CODE

    return out
