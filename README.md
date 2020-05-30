# Person-Vehicle-Bicycle-Detection

REQUIRED: Openvino 2020.1

Person-vehicle-bike detection model is based on SSD detection architecture, RMNet backbone, and learnable image downscale block. The model works in a variety of scenes and weather/lighting conditions. Vehicles bounding boxes are green, pedestrians are blue and bicycles are red. This is user configurable.

Install / Config

Copy all the files to the same folder. python3 app.py -h to see switches. By default the app will look for stck.mkv unless the user modifys (-i)

see gif for preview

usage: Run inference on an input video [-h] -m M [-i I] [-d D] [-t T] [-c C]

required arguments:
  -m M  The location of the model XML file

optional arguments:
  -i I  The location of the input file
  -d D  The device name, if not 'CPU'
  -t T  Confidence threshold
  -c C  Color of bounding boxes, BL, GR or RD

