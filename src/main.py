from typing import Any
import cv2
import numpy as np

class FlowState:
    frame_y : np.ndarray
    features : np

inpath = r"labeled/0.hevc"

invc = cv2.VideoCapture(inpath)

state_last : FlowState = None

while invc.isOpened():

    ret, frame_bgr = invc.read()

    if not ret: break

    state_now = FlowState()
    if state_last is None: state_last = state_now

    state_now.frame_y = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    state_now.features = cv2.goodFeaturesToTrack(state_now.frame_y, 500, 0.3, 7)

    flow = cv2.calcOpticalFlowPyrLK(state_last.frame_y, state_now.frame_y, state_last.features, state_now.features)

    print(flow)

    # dense optical flow is sloww
    """flow_viz = np.zeros_like(frame_bgr)

    flow = cv2.calcOpticalFlowFarneback(
        state_last.frame_y, state_now.frame_y,
        None,
        0.7, 3, 11, 3, 5, 1.1, 0)
    
    flow_r, flow_theta = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    flow_viz[..., 0] = flow_theta * 180 / np.pi / 2
    flow_viz[..., 1] = 255
    flow_viz[..., 2] = cv2.normalize(flow_r, None, 0, 255, cv2.NORM_MINMAX)

    flow_viz = cv2.cvtColor(flow_viz, cv2.COLOR_HSV2BGR)"""
    
    cv2.imshow("frame_y", state_now.frame_y)
    #cv2.imshow("frame_flow", flow_viz)

    cv2.waitKey(1)

    state_last = state_now
