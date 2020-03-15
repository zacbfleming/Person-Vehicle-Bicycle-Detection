import argparse
import cv2
from inference import Network 
import numpy as np
import os
from openvino.inference_engine import IENetwork, IECore, IEPlugin, IENetLayer, InferRequest


INPUT_STREAM = "/home/artichoke/A_Udacity/Intel_Edge_AI/videoInput/stock.mkv"
CPU_EXTENSION = "/opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so "

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    c_desc = 'Color of bounding boxes, BL, GR or RD'
    t_desc = "Confidence threshold"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Input arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-t", help=t_desc, default= float(0.16))
    optional.add_argument("-c", help=c_desc, default='BL')
    args = parser.parse_args()
    return args

def preprocess(input_image):
    height, width = 1024,1024
    clr = 3
    image = cv2.resize(input_image, (height, width))
    image = image.transpose((2,0,1)) 
    image = image.reshape(1, clr, height, width)
    print(image.shape)
    return(image)



def infer_on_video(args):
    box = [ ]
    box1 = [ ]
    box2 = [ ]
    model = args.m
    if args.c == 'BL':
        box = [int(255), int(0), int(0)]
        box1 = [0, 255, 0]
        box2 = [0, 0, 255]
    elif args.c == 'RD':
        box = [0, 0, 255]
        box1 = [255, 0, 0]
        box2 = [0, 255, 0]
    elif args.c == 'GR':
        box = [0, 255, 0]
        box1 = [0, 0, 255]
        box2 = [255, 0, 0]
    else:
        box = [255, 0, 0]
        box1 = [0, 255, 0]
        box2 = [0, 0, 255]
    print(box[0], box1[1], box2[:])

### Setup video file to be written
    cap = cv2.VideoCapture(args.i, cv2.CAP_FFMPEG)
    cap.open(args.i)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
    out = cv2.VideoWriter('vid.avi', fourcc, 30, (width,height))


### Open vid and read
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        
        key_pressed = cv2.waitKey(30)
### Process each frame
        image = preprocess(frame)     
        exec_net = Network.load_model(model,image,args.d, CPU_EXTENSION)
        input_blob = next(iter(exec_net.inputs))
        request = Network.async_inference(exec_net, image, input_blob)
        output_blob = next(iter(request))
        detections = request[0][0][:]
        c = 0
        thresh = float(args.t)
        print(args.c)

### Individual objects are divided into detect and parameters are put into seperate fields
        for detect in detections:
            c+=1
            conf = detect[2]
            label = detect[1]
            rq = detect[0]
### -1 rq indicates the last detection within detections(200 possible)
            if rq == -1:
                c = 0
                break
### Define bounding box shape
            xmin = int(detect[3] * width)
            ymin = int(detect[4]* height)
            xmax = int(detect[5] * width)
            ymax = int(detect[6] * height)
            pone = (xmin, ymin)
            ptwo = (xmax, ymax)
            xdiff = xmax - xmin
            ydiff = ymax - ymin
            df = xdiff * ydiff
            #print(c, detect)
            rn = np.random.rand(3, 1)
            rrn = (int(sum(rn)))
     
### Draw bounding boxes
            if conf >= thresh and label == 1 and df < 1000:
                 cv2.rectangle(frame, pone, ptwo, (int(box[0]), int(box[1]), int(box[2])), 2)
            elif conf >= thresh and label == 2 and df < 2000 and ymin>300:
                cv2.rectangle(frame, pone, ptwo, (box1), 2)
            elif conf >= thresh and label == 3 and df < 1000:
                cv2.rectangle(frame, pone, ptwo, (box2), 2)
                
            else:continue
        cv2.imshow('frame', frame)  
        out.write(frame)
          
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            
        
### Release video
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    
    
def main():
    args = get_args()
    infer_on_video(args)
    #infer_age(args)

if __name__ == "__main__":
    main()
