import numpy as np
from scipy import ndimage as ndi
from skimage import color
from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel, sobel
from tqdm import tqdm


def preprocess_images(images, preprocess_function):
    """
    Process a list of images using a specified preprocessing function to prepare features for each image.

    Args:
        images (list): A list of images to be processed. Each image is expected to be a NumPy array.
        preprocess_function (callable): A function that preprocesses a single image. It should take a single image as input
                                        and return a processed feature vector.

    Returns:
        ndarray: A 2D array where each row is a feature vector corresponding to an image.
    """
    # Process each image in the list using the provided preprocessing function
    processed_images = [
        preprocess_function(img) for img in tqdm(images, desc="Processing images")
    ]

    # Stack the processed image features vertically to create a feature matrix
    X = np.vstack(processed_images)
    return X


def get_sobel_features(image):
    """
    Compute Sobel edge detection on an image.

    This function applies the Sobel filter to an image to highlight edges. It returns the edge
    intensity image which can be used as a feature for further analysis or processing.

    Args:
        image (ndarray): An image array in RGB format.

    Returns:
        ndarray: Edge intensity image derived from applying the Sobel filter.

    Read more about Sobel edge detection: https://en.wikipedia.org/wiki/Sobel_operator and https://scikit-image.org/docs/stable/auto_examples/edges/plot_edge_filter.html
    """
    image_gray = color.rgb2gray(image)
    edges = sobel(image_gray)
    return edges


def generate_gabor_kernel(theta, sigma, frequency):
    """
    Generate a collection of Gabor filter kernels with specified parameters.

    This function creates a list of Gabor kernels each with different orientations and properties
    defined by the input parameter ranges for theta (orientation), sigma (bandwidth), and frequency.

    Args:
        theta_vals (list or array): Range of orientations for the Gabor filter kernels.
        sigma_vals (list or array): Range of sigma values (bandwidths) for the kernels.
        frequency_vals (list or array): Range of frequencies for the kernels.

    Returns:
        list: A list of Gabor kernels with specified parameters.

    Read more about Gabor filters:
    https://en.wikipedia.org/wiki/Gabor_filter and https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_gabor.html
    for more filters: https://scikit-image.org/docs/stable/api/skimage.filters.html
    """

    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
    return kernel


def get_gabor_features(image, kernel):
    """
    Extract features from an image using a single Gabor filter kernel.

    This function applies the provided Gabor kernel to the input image and returns the filtered image.
    The filtered image represents the Gabor features extracted using the specified kernel, which can capture
    specific texture information based on the kernel's parameters.

    Args:
        image (ndarray): An image array in RGB format.
        kernel (ndarray): A single Gabor kernel.

    Returns:
        ndarray: Filtered image representing Gabor features extracted with the kernel.
    """
    image_gray = color.rgb2gray(image)
    filtered = ndi.convolve(image_gray, kernel, mode="wrap")
    return filtered


def get_local_binary_pattern(image, radius=1, method="uniform"):
    """
    Compute the Local Binary Pattern (LBP) descriptor for an image.

    This function calculates the LBP descriptor for the input image. LBP is a powerful texture descriptor
    which can be used for further image analysis or classification. The function returns the LBP image
    that can be further processed or used directly as a feature.

    Args:
        image (ndarray): An image array in RGB format.
        radius (int): Radius of circular LBP. Defaults to 1.
        method (str): Method to compute LBP. Defaults to 'uniform'.

    Returns:
        ndarray: LBP image.

    Read more about LBP: https://en.wikipedia.org/wiki/Local_binary_patterns and https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html
    """
    image_gray = color.rgb2gray(image.astype(np.float32))
    n_points = 8 * radius
    lbp = local_binary_pattern(image_gray.astype(np.uint8), n_points, radius, method)
    return lbp
