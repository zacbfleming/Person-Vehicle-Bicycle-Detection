import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin, IENetLayer, InferRequest

cpu_extension = "/opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so "

class Network:

    def __init__(self):
        self.plugin = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(model, image, device, cpu_extension):
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        net = IENetwork(model=model_xml, weights=model_bin)
        plugin = IECore()
        exec_net = plugin.load_network(network=net, device_name='CPU', num_requests=2)
        return exec_net

    
    def async_inference(self, image, input_blob):
        infer_request_handle = self.start_async(request_id=0, inputs={input_blob: image})
        request_status = infer_request_handle.wait()
        res = infer_request_handle.outputs['detection_out']
        return res

    
    def extract_output(self, network):
        output_blob = next(iter(network.outputs))
        det = self[output_blob]
        return det
