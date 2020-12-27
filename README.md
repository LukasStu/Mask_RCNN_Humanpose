Mask RCNN for Human Pose Estimation
-----------------------------------
# Project from SuperLee506. Updated to Tensorflow 2.1 and tf.keras
* The original code is from "https://github.com/matterport/Mask_RCNN".
* This code helped me a lot when switching to tf.keras "https://github.com/akTwelve/Mask_RCNN".


# My Environment on Windows 10
* conda create -n tf21 python=3.7 git numpy scikit-image scipy Pillow cython h5py pandas opencv pydot tensorflow-gpu==2.1.0
* pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI


# Getting Started
* [inference_humanpose.ipynb][5] shows how to predict the keypoint of human using my trained model. It randomly chooses a image from the validation set. You can download pre-trained COCO weights for human pose estimation (mask_rcnn_coco_humanpose.h5) from the releases page (https://github.com/Superlee506/Mask_RCNN_Humanpose/releases).
* [train_humanpose.ipynb][6] shows how to train the model step by step. You can also use "python train_humanpose.py" to  start training.
* [inspect_humanpose.ipynb][7] visulizes the proposal target keypoints to check it's validity. It also outputs some innner layers to help us debug the model.
* [demo_human_pose.ipynb][8] A new demo for images input from the "images" folder. [04-11-2018]
* [video_demo.py][9] A new demo for video input from camera.[04-11-2018]

# Discussion (Superlee506)
* I convert the joint coordinates into an integer label ([0, 56*56)), and use  `tf.nn.sparse_softmax_cross_entropy_with_logits` as the loss function. This refers to the original [Detectron code][10] which is key reason why my loss can converge quickly.
* If you still want to use the keypoint mask as output, you'd better adopt the modified loss function proposed by [@QtSignalProcessing][11] in [issue#2][12]. Because after crop and resize, the keypoint masks may hava more than one 1 values, and this will make the original soft_cross entropy_loss hard to converge.
* Althougth the loss converge quickly, the prediction results isn't as good as the oringal papers, especially for right or left shoulder, right or left knee, etc. I'm confused with it, so I release the code and any contribution or suggestion to this repository is welcome.


  [1]: https://github.com/matterport/Mask_RCNN/issues/2
  [2]: https://github.com/RodrigoGantier/Mask_R_CNN_Keypoints
  [3]: https://github.com/matterport/Mask_RCNN
  [4]: https://github.com/RodrigoGantier/Mask_R_CNN_Keypoints/issues/3
  [5]: https://github.com/Superlee506/Mask_RCNN/blob/master/inference_humanpose.ipynb
  [6]: https://github.com/Superlee506/Mask_RCNN/blob/master/train_human_pose.ipynb
  [7]: https://github.com/Superlee506/Mask_RCNN/blob/master/inspect_humanpose.ipynb
  [8]: https://github.com/Superlee506/Mask_RCNN_Humanpose/blob/master/demo_human_pose.ipynb
  [9]: https://github.com/Superlee506/Mask_RCNN_Humanpose/blob/master/video_demo.py
  [10]: https://github.com/facebookresearch/Detectron/blob/master/lib/utils/keypoints.py
  [11]: https://github.com/QtSignalProcessing
  [12]: https://github.com/matterport/Mask_RCNN/issues/2
