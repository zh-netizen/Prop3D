import os

import dotenv as de  # For loading environment variables from a .env file
import numpy as np
import pandas as pd

from atom3d.datasets import LMDBDataset  # Dataset class to load LMDB format 3D atomic data
from atom3d.util.voxelize import dotdict, get_center, gen_rot_matrix, get_grid  # Utility functions for voxelization
from torch.utils.data import DataLoader  # PyTorch DataLoader for batching


# Load environment variables, looking for a .env file in current working directory
de.load_dotenv(de.find_dotenv(usecwd=True))


class CNN3D_TransformSMP(object):
    def __init__(self, label_name, random_seed=None, **kwargs):
        """
        Initialize transformation with label to predict and voxel grid parameters
        """
        self.label_name = label_name  # Name of the label to extract from dataset
        self.random_seed = random_seed  # Seed for random rotation matrix (for reproducibility)

        # Default voxel grid configuration
        self.grid_config = dotdict({
            # Mapping atom elements to voxel channels
            'element_mapping': {
                'H': 0,
                'C': 1,
                'O': 2,
                'N': 3,
                'F': 4,
            },
            # Radius (in angstroms) of the voxel grid
            'radius': 7.5,
            # Resolution (voxel size in angstroms)
            'resolution': 1.0,
            # Number of directions for rotational data augmentation
            'num_directions': 20,
            # Number of rolls (rotations) for data augmentation
            'num_rolls': 20,
        })
        # Update default grid config with any additional keyword arguments
        self.grid_config.update(kwargs)

    def _voxelize(self, atoms):
        """
        Convert atomic coordinates and elements into a 3D voxel grid representation.
        """
        # Extract atom positions as float32 numpy array
        pos = atoms[['x', 'y', 'z']].astype(np.float32)
        # Compute the geometric center of the molecule to center the voxel grid
        center = get_center(pos)
        # Generate a random rotation matrix for data augmentation
        rot_mat = gen_rot_matrix(self.grid_config, random_seed=self.random_seed)
        # Generate voxel grid of atom positions with given rotation and config
        grid = get_grid(atoms, center, config=self.grid_config, rot_mat=rot_mat)
        # Move the channel dimension (atom types) to the first axis for PyTorch compatibility
        grid = np.moveaxis(grid, -1, 0)
        return grid

    def _lookup_label(self, item, name):
        """
        Retrieve the target label value from the item’s label dictionary by name.
        """
        # Initialize label mapping dictionary on first call
        if 'label_mapping' not in self.__dict__:
            # List of possible label names in the SMP dataset
            label_mapping = [
                'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
                'u0', 'u298', 'h298', 'g298', 'cv',
                'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom', 'cv_atom',
            ]
            # Create dictionary: label_name -> index
            self.label_mapping = {k: v for v, k in enumerate(label_mapping)}
        # Return the label value corresponding to the requested name
        return item['labels'][self.label_mapping[name]]

    def __call__(self, item):
        """
        Callable method to transform a dataset item:
        - Convert molecule to voxel grid
        - Extract label value
        - Return a dictionary with features, label, and id
        """
        transformed = {
            'feature': self._voxelize(item['atoms']),  # Voxelized atomic features
            'label': self._lookup_label(item, self.label_name),  # Target label value
            'id': item['id'],  # Molecule unique identifier
        }
        return transformed


if __name__ == "__main__":
    # Construct path to the validation set folder using environment variable SMP_DATA
    dataset_path = os.path.join(os.environ['SMP_DATA'], 'val')

    # Load LMDB dataset with the voxelization transform, increasing radius to 10 angstroms
    dataset = LMDBDataset(
        dataset_path,
        transform=CNN3D_TransformSMP(label_name='alpha', radius=10.0)
    )

    # Create PyTorch DataLoader for batching dataset items
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Iterate over DataLoader and print shape and sample labels for the first batch only
    for item in dataloader:
        print('feature shape:', item['feature'].shape)  # Shape of voxelized feature tensor
        print('label:', item['label'])  # Corresponding label values for the batch
        print('id:', item['id'])  # IDs of molecules in the batch
        break  # Exit after first batch for demonstration
