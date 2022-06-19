import pixellib
import os
from pixellib.instance import instance_segmentation
import cv2

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
segmentation_model = instance_segmentation()

segmentation_model.load_model(r'C:\Users\diart\PycharmProjects\OpenCV+Pixellib\mask_rcnn_coco.h5')

camera = cv2.VideoCapture(0)

seg_result = segmentation_model.process_camera(
    camera, show_bboxes=True,
    show_frames=True,
    extract_segmented_objects=False,
    save_extracted_objects=False,
    frames_per_second=5,
    frame_name='DiarTorebek',
    output_video_name='output.mp4'

)
