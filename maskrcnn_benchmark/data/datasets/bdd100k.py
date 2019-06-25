import os
import sys
import glob
from PIL import Image
import numpy as np
import torch
import torchvision
import random
import json

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.segmentation_mask import BinaryMaskList



class BDD100kDataset(torch.utils.data.Dataset):
#class BDD100Dataset():
	"""
		A generic Dataset for the maskrcnn_benchmark must have the following
		non-trivial fields / methods implemented:
		classid_to_name - dict:
		This will allow the trivial generation of classid_to_ccid
		(contiguous) and ccid_to_classid (reversed)
		_getitem__ - function(idx):
		This has to return three things: img, target, idx.
		img is the input image, which has to be load as a PIL Image object
		implementing the target requires the most effort, since it must have
		multiple fields: the size, bounding boxes, labels (contiguous), and
		masks (either COCO-style Polygons, RLE or torch BinaryMask).
		Ideally the target is a BoxList instance with extra fields.
		Lastly, idx is simply the input argument of the function.
		"""
	def __init__(self,root, ann_file, transforms=None):
		"""
		Arguments:
		ann_file: path to file with annotations
		root: path to file with images
		transform:  function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.ToTensor``
		"""
		sys.path.append(root)
		sys.path.append(ann_file)
		self.imgs = self.extract_images(root)
		self.labels = self.extract_labels(ann_file)			
		self.masks = SegmentationMask(self.extract_masks(ann_file + '/rleData'))
		
	"""
	Given filepath, pull all RLE encodings of the given file and return as a BinaryMaskList
	"""
	def extract_masks(self, rleMaskfilename):
		with open(rleMaskfilename) as f:
			print("OPENED")
			rleMaskfile = f.read()
		return BinaryMaskList(rleMaskfile, self.imgs[0].size())

	"""
	Given filepath, pull all of the images from directory and return as a iterable map
	"""
	def extract_images(self, root):
		print(root)
		images = []
		for filename in os.listdir(root):

			curImage = Image.open(root + filename)
			curImage.close()
			images.append(curImage)
		return images		
		
	"""
	Given filepath, pull of the annotations and return an organized a list of dictionaries, each dictionary containing a label
	"""
	def extract_labels(self, label_file):
		print(label_file)
		labels = []
		try:
			with open(label_file) as f:
				print("OPENED")
				stringData = f.read()
				data = json.loads(stringData)
				#print(json.dumps(data[0], sort_keys = True, indent = 4))
				f.close()
				return data
		except IOError:
			print("Error: File does not exist")
			return 0



	def __getitem__(self, index):
		image = self.imgs[index]
		curData = self.labels[index]
		labelList = []
		boxList = []
		masks = self.masks[index]
		for key in curData["labels"]:
			if key["category"] != "drivable area" and key["category"] != "lane":
				labelList.append(key["category"])
				box = key["box2d"]["x1"], key["box2d"]["y1"], key["box2d"]["x2"], key["box2d"]["y2"]
				boxList.append(box)
			else:
				poly = key["poly2d"]

				for vertices in poly:
					vert1 = [int(vertices["vertices"][0][0]), int(vertices["vertices"][0][1])]
					vert2 = vert1
					for point in vertices["vertices"]:
						if (vert1[0] > int(point[0]) and vert1[1] > int(point[1])):
							vert1 = point
						if (vert2[0] > int(point[0]) and vert2[1] > int(point[1])):
							vert2 = point
				box = vert1[0], vert1[1], vert2[0], vert2[1]
				boxList.append(box)

		
		
		target = BoxList(boxList, image.size, mode="xyxy")
		target.add_field("labels", labelList)
		masks = SegmentationMask(masks, image.size, "mask")
		target.add_field("masks", masks)
		return image, target, index
		
	def get_img_info(self, index):
		image = self.imgs[index]
		return {"height": image.height, "width": image.width}


	def __len__(self):
		return(len(self.imgs))
