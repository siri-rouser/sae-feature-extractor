import logging
import cv2
import torch
import time
import numpy as np
from typing import Any, Dict, NamedTuple
from PIL import Image
import io
from visionlib.pipeline.tools import get_raw_frame_data
from prometheus_client import Counter, Histogram, Summary
from feature_extractor.vision_api.python.visionapi.visionapi.messages_pb2 import BoundingBox, SaeMessage
from .utils import ReidFeature

from .config import FeatureExtrator

logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')
logger = logging.getLogger(__name__)

GET_DURATION = Histogram('my_stage_get_duration', 'The time it takes to deserialize the proto until returning the tranformed result as a serialized proto',
                         buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))
OBJECT_COUNTER = Counter('my_stage_object_counter', 'How many detections have been transformed')
PROTO_SERIALIZATION_DURATION = Summary('my_stage_proto_serialization_duration', 'The time it takes to create a serialized output proto')
PROTO_DESERIALIZATION_DURATION = Summary('my_stage_proto_deserialization_duration', 'The time it takes to deserialize an input proto')


class Extractor:
    def __init__(self, config: FeatureExtrator) -> None:
        self.config = config
        logger.setLevel(self.config.log_level.value)

        self._setup()

    def __call__(self, input_proto) -> Any:
        return self.get(input_proto)
    
    @GET_DURATION.time()
    @torch.no_grad()
    def get(self, input_proto):
        input_image, sae_msg = self._unpack_proto(input_proto)
        # for the input_image: <class 'numpy.ndarray'>, size: 1080 bytes

        # Your implementation goes (mostly) here
        inference_start = time.monotonic_ns()
        det_array = self._prepare_detection_input(sae_msg)
        # logger.warning('Received SAE message from pipeline')
        img_list = self._imglist(input_image,det_array)

        if not img_list:
            raise ValueError("No valid images to process.")


        feature_output = self.model.extract(img_list)
        inference_time_us = (time.monotonic_ns() - inference_start) // 1000


        return self._create_output(feature_output, det_array, sae_msg, inference_time_us)
        
    @PROTO_DESERIALIZATION_DURATION.time()
    def _unpack_proto(self, sae_message_bytes):
        sae_msg = SaeMessage()
        sae_msg.ParseFromString(sae_message_bytes)

        input_frame = sae_msg.frame
        input_image = get_raw_frame_data(input_frame)

        # Log the type and size of the input image to debug
        # logger.debug(f'Input image type: {type(input_image)}, size: {len(input_image)} bytes')

        return input_image, sae_msg
    
    @PROTO_SERIALIZATION_DURATION.time()
    def _pack_proto(self, sae_msg: SaeMessage):

        
        return sae_msg.SerializeToString()
    
    def _setup(self):
        self.model = ReidFeature(self.config)

    def _prepare_detection_input(self, sae_msg: SaeMessage):
        det_array = np.zeros((len(sae_msg.detections), 6))
        for idx, detection in enumerate(sae_msg.detections):
            det_array[idx, 0] = detection.bounding_box.min_x
            det_array[idx, 1] = detection.bounding_box.min_y
            det_array[idx, 2] = detection.bounding_box.max_x
            det_array[idx, 3] = detection.bounding_box.max_y

            det_array[idx, 4] = detection.confidence
            det_array[idx, 5] = detection.class_id
        return det_array
    
    def _imglist(self, image_array, dets):
        is_success, buffer = cv2.imencode(".jpg", image_array)
        if not is_success:
            logger.error('Failed to convert NumPy array to bytes')
            raise ValueError('Failed to convert NumPy array to bytes')
        
        image_bytes = io.BytesIO(buffer)

        # Load the image from bytes
        try:
            image = Image.open(image_bytes).convert('RGB')
        except Exception as e:
            logger.error(f'Error opening image: {e}')
            raise

        cropped_images = []

        for idx in range(dets.shape[0]):
            # Extract bounding box coordinates
            min_x, min_y, max_x, max_y = dets[idx, :4]

            # Crop the image
            cropped_image = image.crop((min_x, min_y, max_x, max_y))
            cropped_images.append(cropped_image)

        return cropped_images
    
    @PROTO_SERIALIZATION_DURATION.time()
    def _create_output(self, feature_output, det_array, sae_msg: SaeMessage, inference_time_us):

        if feature_output is None:
            logger.info('No feature extracted')
        else:
            logger.info(f'Feature extracted with shape: {len(feature_output)}')
        
        for idx, detection in enumerate(sae_msg.detections):
            detection.feature.extend(feature_output[idx].tolist())

        sae_msg.metrics.feature_extraction_time_us = inference_time_us
        
        return sae_msg.SerializeToString()

##### assign feature into each detection!!!