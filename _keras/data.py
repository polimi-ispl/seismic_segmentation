from tensorflow.keras.utils import Sequence
import numpy as np
from utils.processing import compute_edges, to_categorical, from_categorical
import pandas as pd


# -------------------- KERAS SEQUENCE ----------------------
class PatchArrayLoader(Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""
    
    def __init__(self, batch_size=32, inputs=None, targets=None,
                 augmentation=None, preprocessing=None,
                 separate_classes=False, use_edges=False, use_inputs=False):
        
        assert len(inputs) == len(targets), "the numbers of inputs and targets mismatch"
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.input_array = inputs
        self.target_array = targets
        self.separate_class = separate_classes

        self.use_edges = use_edges
        self.use_inputs = use_inputs
    
    @property
    def batch_element_idx_stop(self):
        return self.batch_element_idx_start + self.batch_size
    
    def __len__(self):
        """Number of batches"""
        return self.input_array.shape[0] // self.batch_size
        
    def __getitem__(self, batch_idx):
        """Returns tuple (input, target) correspond to batch idx."""
        # compute indexes
        self.batch_element_idx_start = batch_idx * self.batch_size
        
        # create batch elements
        input_batch = self.input_array[self.batch_element_idx_start:self.batch_element_idx_stop]
        target_batch = self.target_array[self.batch_element_idx_start:self.batch_element_idx_stop]
        
        # apply transforms
        for i, (image, mask) in enumerate(zip(input_batch, target_batch)):
            sample = {'image':image, 'mask':mask}
            if self.preprocessing is not None:
                sample = self.preprocessing(sample)
            if self.augmentation is not None:
                sample = self.augmentation(sample)
            input_batch[i] = sample['image']
            target_batch[i] = sample['mask']
            
        if self.use_edges:
            edges = compute_edges(target_batch)
        
        # create a list of batches for targets
        if self.separate_class:
            target_batch = [target_batch[:, :, :, c][:, :, :, None] for c in range(target_batch.shape[-1])]
        else:
            target_batch = [target_batch]
        
        if self.use_edges:
            target_batch.append(edges)
        if self.use_inputs:
            target_batch.append(input_batch)
        
        return input_batch, target_batch


class SectionFileLoader(Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""
    
    def __init__(self, batch_size=32, inputs=None, targets=None,
                 input_transform=None, target_transform=None,
                 separate_class=False, use_edges=False, use_inputs=False):
        
        assert len(inputs) == len(targets), "the numbers of inputs and targets mismatch"
        self.batch_size = batch_size
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.input_file_list = inputs
        self.target_file_list = targets
        self.separate_class = separate_class
        
        self.batch_elements_per_list_element = np.load(inputs[0]).shape[0]
        
        self.input_array_list = []
        self.target_array_list = []
        self.file_idx = 0
        self.batch_element_idx_start = 0
        self.use_edges = use_edges
        self.use_inputs = use_inputs
    
    @property
    def batch_element_idx_stop(self):
        return self.batch_element_idx_start + self.batch_size
    
    def __len__(self):
        """Number of batches"""
        return len(self.input_file_list) * self.batch_elements_per_list_element // self.batch_size
    
    def _load(self, file_idx):
        """Load patch arrays from files and return them as lists"""
        input_array = np.load(self.input_file_list[file_idx])
        target_array = np.load(self.target_file_list[file_idx])
        input_array_list = [input_array[e] for e in range(input_array.shape[0])]
        target_array_list = [target_array[e] for e in range(target_array.shape[0])]
        return input_array_list, target_array_list
    
    def __getitem__(self, batch_idx):
        """Returns tuple (input, target) correspond to batch idx."""
        # compute indexes
        self.batch_element_idx_start = batch_idx * self.batch_size
        
        # add elements to the lists if the already present ones are not sufficient
        if len(self.input_array_list) < self.batch_element_idx_stop:
            if batch_idx != 0:
                self.file_idx += 1
            input_list, target_list = self._load(self.file_idx)
            self.input_array_list += input_list
            self.target_array_list += target_list
        
        # create batch elements
        input_batch = np.asarray(self.input_array_list[self.batch_element_idx_start:self.batch_element_idx_stop])
        target_batch = np.asarray(self.target_array_list[self.batch_element_idx_start:self.batch_element_idx_stop])
        
        # apply transforms
        if self.input_transform is not None:
            input_batch = self.input_transform(input_batch)
        if self.target_transform is not None:
            target_batch = self.target_transform(target_batch)
        
        if self.use_edges:
            edges = compute_edges(target_batch)
        
        # create a list of batches for targets
        if self.separate_class:
            target_batch = [target_batch[:, :, :, c][:, :, :, None] for c in range(target_batch.shape[-1])]
        else:
            target_batch = [target_batch]
        
        if self.use_edges:
            target_batch.append(edges)
        if self.use_inputs:
            target_batch.append(input_batch)
        
        return input_batch, target_batch


class PatchFileLoader(Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""
    
    def __init__(self, batch_size=32, inputs=None,
                 input_transform=None, target_transform=None,
                 separate_class=False, use_edges=False, use_inputs=False):
        
        self.batch_size = batch_size
        
        self.input_transform = input_transform
        self.target_transform = target_transform
        
        self.input_file_list = inputs
        
        self.separate_class = separate_class
        self.use_edges = use_edges
        self.use_inputs = use_inputs
    
    def __len__(self):
        """Number of batches"""
        return len(self.input_file_list) // self.batch_size
    
    def _load(self, file_path):
        """Load patch arrays from files and return them as lists"""
        dataset = np.load(file_path, allow_pickle=True).item()
        return dataset['img'], dataset['msk']
    
    def __getitem__(self, batch_idx):
        """Returns tuple (input, target) correspond to batch idx."""
        # compute indexes
        batch_files = self.input_file_list[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]
        
        # create batch elements
        input_batch = []
        target_batch = []
        for file_idx in batch_files:
            i, m = self._load(file_idx)
            input_batch.append(i)
            target_batch.append(m)
        input_batch = np.asarray(input_batch)
        target_batch = np.asarray(target_batch)
        
        # apply transforms
        if self.input_transform is not None:
            input_batch = self.input_transform(input_batch)
        if self.target_transform is not None:
            target_batch = self.target_transform(target_batch)
        
        if self.use_edges:
            edges = compute_edges(target_batch)
        
        # create a list of batches for targets
        if self.separate_class:
            target_batch = [target_batch[:, :, :, c][:, :, :, None] for c in range(target_batch.shape[-1])]
        else:
            target_batch = [target_batch]
        
        if self.use_edges:
            target_batch.append(edges)
        if self.use_inputs:
            target_batch.append(input_batch)
        
        return input_batch, target_batch


class CSVLoader(Sequence):
    
    def __init__(self, csv_file, phase, batch_size=32, shuffle=False,
                 augmentation=None, preprocessing=None, categorical=False,
                 separate_classes=False, normalize_classes=False, use_edges=False, use_inputs=False, num_classes=1):
        """
        Use a CSV file for defining the lists of patches

        :param csv_file (str): path to file
        :param phase (str): one of train, val, test
        :param batch_size (int): number of batch elements
        :param augmentation (albumentations.Compose): sample augmentation pipeline
        :param preprocessing (albumentations.Compose): sample transformation pipeline
        :param categorical (bool): use the one-hot representation
        :param separate_classes (bool): separate the labels in a list of target elements (one-hot)
        :param normalize_classes (bool): normalize the classes into [0,1] range
        :param use_edges (bool): returns the edges of the labels a target element
        :param use_inputs (bool): returns a copy of the input as a target element
        :param num_classes (int): number of target classes, used for normalization
        """

        df = pd.read_csv(csv_file, converters={'name': str})
        
        self.file_paths = df[df['mode'] == phase]['datapath'].values
        self.indexes = np.arange(len(self.file_paths))
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if separate_classes:
            categorical = True
        if not categorical:
            normalize_classes = True
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.categorical = categorical
        self.separate_classes = separate_classes
        self.use_edges = use_edges
        self.use_inputs = use_inputs
        self.num_classes = num_classes
    
    def __len__(self):
        """Number of batches"""
        return len(self.file_paths) // self.batch_size
    
    def _on_epoch_end(self):
        self.indexes = np.arange(len(self.sample_files_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _load(self, file_path):
        """Load patch arrays from files and return them as lists"""
        return np.load(file_path, allow_pickle=True).item()
    
    def __getitem__(self, batch_idx):
        """Returns tuple (input, target) correspond to batch idx."""
        # compute indexes
        batch_samples_idx = self.indexes[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

        # create batch elements
        input_batch = []
        target_batch = []
        for file_idx in batch_samples_idx:
            sample = self._load(self.file_paths[file_idx])
            if self.preprocessing is not None:
                sample = self.preprocessing(**sample)
            if self.augmentation is not None:
                sample = self.augmentation(**sample)
            input_batch.append(sample['image'])
            target_batch.append(sample['mask'])
        
        input_batch = np.asarray(input_batch)
        target_batch = np.asarray(target_batch)
        
        if self.use_edges:
            edges = compute_edges(target_batch)
        
        # create a list of batches for targets
        if self.separate_classes:  # a target element for every one-hot class
            if target_batch.shape[-1] != 1:
                target_batch = from_categorical(target_batch)
            target_batch = to_categorical(target_batch)
            target_batch = [np.expand_dims(target_batch[:, :, :, c], -1)
                            for c in range(target_batch.shape[-1])]
        elif self.categorical:  # from integers to one-hot classes
            target_batch = [to_categorical(target_batch)]
        else:  # just keep the int targets
            if 'float' in self.file_paths[0]:
                # masks are integers from 0 to num_classes-1, we normalize in [0,1] for sigmoid
                target_batch = [target_batch / (self.num_classes-1)]
            else:
                # from categorical to sparse, then normalized in [0,1] for sigmoid
                target_batch = [np.expand_dims(from_categorical(target_batch), -1) / (self.num_classes-1)]
        
        if self.use_edges:
            target_batch.append(edges)
        if self.use_inputs:
            target_batch.append(input_batch)
        
        return input_batch, target_batch


if __name__ == "__main__":
    
    dataset = "/nas/home/fpicetti/datasets/seismic_facies/train1.csv"
    
    gen = CSVLoader(
        csv_file=dataset,
        phase='train',
        batch_size=4,
        separate_classes=False,
        normalize_classes=True,
        use_edges=False,
        use_inputs=False
    )
    
    for batch_idx in range(len(gen)):
        inputs, targets = gen[batch_idx]
        
        print("batch %s yields %s - %s"
              % (str(batch_idx).zfill(3), str(inputs.shape), str([str(t.shape) for t in targets])))
    
    print(0)
