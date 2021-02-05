def build_fpn_keypoint_graph(rois, feature_maps,
                         image_shape, pool_size, num_keypoints):
    """Builds the computation graph of the keypoint head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_keypoints: number of keypoints, which determines the depth of the results,
    in coco, it's 17, in AI challenger, it's 14

    Returns: keypoint_masks [batch, roi_count, num_keypoints, height*width]
    """

    # ROI Pooling
    # Shape: [batch, num_rois, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name="roi_align_keypoint_mask")([rois] + feature_maps)
    # Eight consecutive convs
    for i in range(8):
        x = KL.TimeDistributed(KL.Conv2D(512, (3, 3), padding="same"),
                               name="mrcnn_keypoint_mask_conv{}".format(i + 1))(x)

        x = KL.TimeDistributed(BatchNorm(axis=3),
                               name='mrcnn_keypoint_mask_bn{}'.format(i + 1))(x)
        x = KL.Activation('relu')(x)
    # Deconvolution
    x = KL.TimeDistributed(KL.Conv2DTranspose(num_keypoints, (2, 2), strides=2),
                           name="mrcnn_keypoint_mask_deconv")(x)
    x = KL.TimeDistributed(
        KL.Lambda(lambda z: tf.image.resize(z, [28, 28], method='bilinear')),name="mrcnn_keypoint_mask_upsample_1")(x)

    #shape: batch_size, num_roi, 56, 56, num_keypoint
    x = KL.TimeDistributed(
        KL.Lambda(lambda z: tf.image.resize(z, [56, 56], method='bilinear')),name="mrcnn_keypoint_mask_upsample_2")(x)
    # shape: batch_size, num_roi, num_keypoint, 56, 56
    x = KL.TimeDistributed(KL.Lambda(lambda x: tf.transpose(x,[0,3,1,2])), name="mrcnn_keypoint_mask_transpose")(x)
    s = K.int_shape(x)
    
    if s[1] is None:
        x = KL.Reshape((-1, num_keypoints, -1), name='mrcnn_keypoint_mask_reshape')(x)
    else:
        x = KL.Reshape((s[1], num_keypoints, -1), name='mrcnn_keypoint_mask_reshape')(x)
    return x