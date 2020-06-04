import cv2

def encode(reid, face):
    try:
        cur_request_id = 0
        in_frame = cv2.resize(face, (reid[3], reid[2]))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((reid[0], reid[1], reid[2], reid[3]))
        reid[4].start_async(request_id=cur_request_id, inputs={reid[5]: in_frame})

        if reid[4].requests[cur_request_id].wait(-1) == 0:
            # Parse detection results of the current request
            res = reid[4].requests[cur_request_id].outputs[reid[6]]
            res = res.reshape(1, 256)
            return res
    except Exception as err:
        print('Exception from {create_encoding()}: ', str(err))
