import numpy as np
import os
import torch

from tiatoolbox.models.dataset import abc
from tiatoolbox.utils.misc import read_locations
from tiatoolbox.tools.patchextraction import get_patch_extractor


class InteractiveSegmentorDataset(abc.PatchDatasetABC):

    def __init__(self, img_path, points, mode, labels=None, IOConfig=None):
        """Creates an interactive segmentation dataset, which inherits from the
            torch.utils.data.Dataset class.

        Args:
            img_path (:obj:`str` or :obj:`pathlib.Path`): Path to a standard image,
                a whole-slide image or a large tile to read.
            points (ndarray, pd.DataFrame, str, pathlib.Path): Points ('clicks') for the image. 
            labels: List of label for sample at the same index in `inputs`.
                Default is `None`.
            mode (str): Type of the image to process. Choose from either `patch`, `tile`
                or `wsi`.

        Examples:
            >>> # an user defined preproc func and expected behavior
            >>> preproc_func = lambda img: img/2  # reduce intensity by half
            >>> transformed_img = preproc_func(img)
            >>> # create a dataset to get patches preprocessed by the above function
            >>> ds = InteractiveSegmentorDataset(
            ...     img_path = 'example_image.png',
            ...     points = 'example_points.csv',
            ...     mode = 'patch',         
            ... )

        """
        # Not using IOConfig at the moment

        super().__init__()

        if not os.path.isfile(img_path):
            raise ValueError("`img_path` must be a valid file path.")
        if mode not in ["patch", "wsi", "tile"]:
            raise ValueError(f"`{mode}` is not supported.")

        self.img_path = img_path
        self.labels = labels
        self.mode = mode


        # TODO: Change to use IOConfig
        self.patch_size = (128,128)     # bounding box size
        self.resolution = 0
        self.units = "level"

        # Read the points('clicks') into a panda df
        self.locations = read_locations(points)


        self.patch_extractor = get_patch_extractor("point",  
            input_img = self.img_path, locations_list = points, patch_size=self.patch_size,
            resolution = self.resolution, units = self.units)


    def __getitem__(self, idx):
        patch = self.patch_extractor.__getitem__(idx)

        boundingBox = self.get_boundingBox(idx)

        # we know nucPoint is at the centre of the patch:
        nucPoint = np.zeros((1, self.patch_size[1], self.patch_size[1]), dtype=np.uint8)
        nucPoint[0,int((self.patch_size[1]-1)/2),int((self.patch_size[1]-1)/2)] = 1

        exclusionMap = self.get_exclusionMap(idx, boundingBox)

        patch = np.moveaxis(patch, 2, 0)
        patch = patch / 255

        input = np.concatenate((patch, nucPoint, exclusionMap), axis=0, dtype=np.float32)   # shape=(c=5,h,w)
 
        data = {
            "image": input,
            "boundingBox": boundingBox,
            "click": (self.locations["x"][idx], self.locations["y"][idx])
        }
        if self.labels is not None:
            data["label"] = self.labels[idx]
        
        return data


    def get_boundingBox(self, idx):
        """This function returns a bounding box of size (patch_size x patch_size) that has the click as its centre.
            The bounding box is the same box that is used in patch extraction.

        Args:
            idx (int): The index of the point ("Click") to get a bounding box for.
        Returns:
            bounds: a list of coordinates in `[start_x, start_y, end_x, end_y]`
            format to be used for patch extraction.

        """

        #Coordinates of the top left corner of each patch:
        location = (self.patch_extractor.locations_df["x"][idx], self.patch_extractor.locations_df["y"][idx])

        tl = np.array(location)
        br = location + np.array(self.patch_size)
        bounds = np.concatenate([tl, br])

        return  bounds

    
    def get_exclusionMap(self, idx, boundingBox):
        """This function returns an exclusion map for click at the given index.

        Args:
            idx (int): The index of the point ("Click") to get an exclusionMap for.
            boundingBox: a list of coordinates in `[start_x, start_y, end_x, end_y]`
                This is the bounding box for the click at the given index.
        Returns:
            exclusionMap (ndarray)

        """

        # Exclude the current click from list of clicks:
        otherPoints = self.locations.drop(idx, axis = 0)
        x_locations = otherPoints["x"].to_numpy()
        y_locations = otherPoints["y"].to_numpy()
      
        exclusionMap = np.zeros((1, self.patch_size[1], self.patch_size[1]), dtype=np.uint8)
        
        xStart = boundingBox[0]
        yStart = boundingBox[1]
        xEnd = boundingBox[2]
        yEnd = boundingBox[3]

        

        for i in range(x_locations.shape[0]):
            x = x_locations[i]
            y = y_locations[i]

            if (x >= xStart and x <= xEnd and y >= yStart and y <= yEnd):
                exclusionMap[0, x - xStart, y - yStart] = 1


        return exclusionMap

    def __len__(self):
        return self.locations.shape[0]       



# df = read_locations("/Users/jlv/Desktop/points/points1.csv")
# cx = df["x"].to_numpy()
# cy = df["y"].to_numpy()
# print(cx, cy)
# print(df["x"][0])
# location = (0,0)
# size = (10,10)
# tl = np.array(location)
# br = location + np.array(size)
# bounds = np.concatenate([tl, br])
# print(bounds)

dataset = InteractiveSegmentorDataset(img_path="/Users/jlv/Desktop/images/image1.png", points="/Users/jlv/Desktop/points/points1.csv",
    mode="patch")
# bb = dataset.get_boundingBox(1)
# exclusionMap = dataset.get_exclusionMap(1, bb)
# print(exclusionMap.shape)
# exclusionMap = np.squeeze(exclusionMap)
# print(exclusionMap.shape)

# plt.imshow(exclusionMap, cmap='Greys_r', interpolation='nearest')
# plt.scatter([64], [64], c='r')
# plt.show()
# box = dataset.get_boundingBox(1)
# print(box)
# data = dataset.__getitem__(4)

# input = data["input"]
# image = input[0:3, :, :]
# image = np.moveaxis(image, 0, 2)

# inclusionMap = input[3, :, :]
# inclusionMap = np.squeeze(inclusionMap)
# click = np.argwhere(inclusionMap>=1)

# exclusionMap = input[4, :, :]
# exclusionMap = np.squeeze(exclusionMap)
# others = np.argwhere(exclusionMap>=1)

# import matplotlib.pyplot as plt
# plt.figure(), plt.imshow(image)
# plt.scatter([click[0][0]],[click[0][1]],c='r')
# plt.scatter(others[:, 0],others[:, 1],c='g')
# plt.show()


dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            drop_last=False,
            shuffle=False,
        )

for _, batch_data in enumerate(dataloader):
    print("load:")
    print(batch_data["input"].shape)



