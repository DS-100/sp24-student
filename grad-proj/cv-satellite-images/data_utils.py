import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def load_images(images_path):
    """
    Load images from a specified .npz file.

    Parameters:
    - images_path (str): The file path to the .npz file containing the images.

    Returns:
    - images (list): A list of numpy arrays, each representing an image loaded from the .npz file.
    """
    data = np.load(images_path, allow_pickle=True)
    images = [data[f"image_{i}"] for i in range(len(data.files))]
    return images

def load_labels(labels_path):
    """
    Load labels from a specified .npy file.

    Parameters:
    - labels_path (str): The file path to the .npy file containing the labels.

    Returns:
    - labels (numpy.ndarray): An array of labels loaded from the .npy file.
    """
    labels = np.load(labels_path, allow_pickle=True)
    return labels

def get_images(data_dir, disaster, split="train"):
    """
    Load images from a specified disaster dataset split.

    Args:
        data_dir (str): The directory where the dataset is stored.
        disaster (str): The disaster type of the dataset.
        split    (str): The train or test split (default train).

    Returns:
        list: A list of images (as numpy arrays) from the specified dataset split.
    """
    images_path = os.path.join(data_dir, disaster, f"{split}_images.npz")
    return load_images(images_path)


def get_labels(data_dir, disaster, split="train"):
    """
    Load labels for a specified disaster dataset split.

    Args:
        data_dir (str): The directory where the dataset is stored.
        disaster (str): The disaster type of the dataset.
        split    (str): The train or test split (default train).

    Returns:
        ndarray: The labels for the images in the specified dataset split.
    """
    labels_path = os.path.join(data_dir, disaster, f"{split}_labels.npy")
    return load_labels(labels_path)


def convert_dtype(images, dtype=np.float32):
    """
    Convert the data type of a collection of images.

    Args:
        images (list or dict): The images to convert, either as a list or dictionary of numpy arrays.
        dtype (data-type): The target data type for the images. Defaults to np.float32.

    Returns:
        The converted collection of images, in the same format (list or dict) as the input.
    """
    if isinstance(images, dict):
        return {k: v.astype(dtype) for k, v in images.items()}
    elif isinstance(images, list):
        return [img.astype(dtype) for img in images]
    else:
        raise TypeError("Unsupported type for images. Expected list or dict.")


def plot_label_distribution(labels, ax=None, title="Label Distribution"):
    """
    Plot the distribution of labels.

    Args:
        labels (ndarray): An array of labels to plot the distribution of.
        ax (matplotlib.axes.Axes, optional): The matplotlib axis on which to plot.
                                             If None, a new figure and axis are created.
        title (str, optional): The title for the plot. Defaults to "Label Distribution".
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True  # Flag indicating a figure was created within this function
    else:
        created_fig = False

    sns.countplot(x=labels, ax=ax, palette="viridis")
    ax.set_title(title)
    ax.set_xlabel("Labels")
    ax.set_ylabel("Count")

    if created_fig:
        plt.tight_layout()
        plt.show()
