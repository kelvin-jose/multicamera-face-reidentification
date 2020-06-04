import os
import sys
from openvino.inference_engine import IENetwork, IEPlugin


def load_xml_bin(model_path):
    model_xml = model_path
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device='CPU', plugin_dirs=None)
    plugin.add_cpu_extension('lib/libcpu_extension_sse4.so')
    # Read IR
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            sys.exit(1)

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    exec_net = plugin.load(network=net, num_requests=2)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    del net
    return n, c, h, w, exec_net, input_blob, out_blob
