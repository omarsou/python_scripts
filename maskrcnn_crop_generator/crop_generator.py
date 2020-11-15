from utils.lazy_object_load import maskrcnn_resnet as model
import cv2
import torchvision.transforms as T

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class ImageDetectorCropError(BaseException):
    """Handle errors."""
    pass


class ImageDetectorCrop:
    def __init__(self, resize=False, size=(224, 224), use_gpu=False):
        """Initialize the ImageDetectorCrop.

        Parameters
        ----------
        resize : bool
                 if True, resize every images that we extract

        size : tuple
               Indicates the size if resize is set to True

        use_gpu : bool
                  Indicates if we use gpu
        """
        self.use_gpu = use_gpu
        self.resize = resize
        self.size = size

    def crop_image(self, img, categories, threshold):
        """Crop the images and return the images corresponding to each bounding boxes of the category in categories.

        Parameters
        ----------
        img : str or ndarray
              if str : path of the image
              if ndarray : Array of shape (height_image, width_image, 3)

        categories : list
                     list of the categories you need to extract from the image

        threshold : float
                    minimum score needed to accept the target extracted by the model

        Returns
        -------
         images : dict
                  Each key is the name of the category that we want and each value is a list of images that correspond
                  to this category
        """
        if set(categories) > set(COCO_INSTANCE_CATEGORY_NAMES):
            missings = [cat for cat in categories if cat not in COCO_INSTANCE_CATEGORY_NAMES]
            raise ImageDetectorCropError(f"{missings} are not in the model category names. Please use only category "
                                         f"that the model can identify.")
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            if self.use_gpu:
                boxes, classes = self.get_prediction_gpu(img, threshold)
            else:
                boxes, classes = self.get_prediction_cpu(img, threshold)
        except IndexError:
            return {v: [] for v in categories}
        return self.get_images(img, boxes, classes, categories)

    def get_images(self, img, boxes, classes, categories):
        """Crop the images and return the images corresponding to each bounding boxes of the category in categories.

        Parameters
        ----------
        img : ndarray
              Array of shape (height_image, width_image, 3)

        boxes : list
                list of the boxes's coordinates

        classes : list
                  list of the corresponding category for each boxes coordinates

        categories : list
                     list of the categories you need to extract from the image

        Returns
        -------
         boxes_categories : dict
                  Each key is the name of the category that we want and each value is a list of images that correspond
                  to this category
        """
        boxes_categories = {v: [] for v in categories}
        for i in range(len(classes)):
            if classes[i] in categories:
                coordinates = boxes[i]
                y, x = coordinates[0], coordinates[1]
                x1, y1, x2, y2 = int(y[0]), int(y[1]), int(x[0]), int(x[1])
                img_crop = img[y1:y2, x1:x2]
                if self.resize:
                    img_crop = cv2.resize(img_crop, self.size, interpolation=cv2.INTER_LANCZOS4)
                boxes_categories[classes[i]].append(img_crop)
        return boxes_categories

    @staticmethod
    def get_prediction_gpu(img, threshold):
        transform = T.Compose([T.ToTensor()])
        img = transform(img).cuda()
        pred = model.value.cuda()([img])
        pred_score = list(pred[0]['scores'].cpu().detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().detach().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return pred_boxes, pred_class

    @staticmethod
    def get_prediction_cpu(img, threshold):
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = model.value([img])
        pred_score = list(pred[0]['scores'])
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'])]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'])]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return pred_boxes, pred_class



