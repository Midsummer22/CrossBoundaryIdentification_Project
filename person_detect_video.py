import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import os
from utils_track import detect_postprocess, preprocess_img, draw_detect_res,process_points,isInsidePolygon
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.augmentations import  letterbox
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync
from track.tracker.byte_tracker import BYTETracker
import numpy as np
from track.tracking_utils.timer import Timer
from track.utils.visualize import plot_tracking
# @torch.no_grad()

track_id_status = {}
points = [[593,176],[904,243],[835,323],[507,259]]
color_light_yellow=(0, 165, 255)   ##浅黄色
color_light_green=(144, 238, 144)  ##浅绿色

def main(opt):
    # Load model
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn, data=opt.data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

    # Half
    opt.half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if opt.half else model.model.float()

    # load image
    cap = cv2.VideoCapture("video_detect.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # video_write = cv2.VideoWriter("video_save.avi", cv2.VideoWriter_fourcc(*'MJPG'), 5, (width, height), isColor=True)

    # load bytetrack
    tracker = BYTETracker(opt, frame_rate=30)
    frame_id = 0
    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            print("Camera cap over!")
            continue
        frame_id += 1
        if not int(frame_id) % 5 == 0: continue

        im0s = img.copy()
        img = letterbox(img, opt.imgsz, stride=32, auto=True)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)
        im = torch.from_numpy(im).to(device)
        im = im.half() if opt.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=opt.augment, visualize=opt.visualize)

        # 如果需要修改跟踪不同的目标，修改non_max_suppression函数中的torch.index_select
        det,output1 = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)

        # 绘制越界监测区域
        im0s = process_points(im0s, points, color_light_green)

        if len(det):
            # Rescale boxes from img_size to im0s size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
            if det[:, :4] is not None:
                online_targets = tracker.update(det, [im0s.shape[0], im0s.shape[1]])
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > opt.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                    im0s = plot_tracking(im0s, online_tlwhs, online_ids, str(round(t.score,1)), frame_id=frame_id + 1,fps=fps)

                    # 判断每个trackid人的状态
                    pt = [tlwh[0] + 1 / 2 * tlwh[2], tlwh[1] + tlwh[3]]
                    if isInsidePolygon(pt, points):
                        cv2.putText(im0s, 'Warning!!!', (int(tlwh[0]), int(tlwh[1])-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 2)
        cv2.imshow("image", im0s)
        cv2.waitKey(10)
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'models/yolov5s.pt', help='model path(s)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
