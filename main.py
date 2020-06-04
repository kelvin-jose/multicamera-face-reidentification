import cv2
from sys import exit
from encoder import encode
from utils.elastic import *
from collections import Counter
from intel_loader import load_xml_bin
from imutils.video import WebcamVideoStream
from elasticsearch.helpers import bulk


def face_detection(frame, face_detection_openvino):
    height, width = frame.shape[:2]
    bboxes = []
    face_in_frame = cv2.resize(frame, (face_detection_openvino[3], face_detection_openvino[2]))
    face_in_frame = face_in_frame.transpose((2, 0, 1))
    face_in_frame = face_in_frame.reshape((face_detection_openvino[0], face_detection_openvino[1],
                                           face_detection_openvino[2], face_detection_openvino[3]))
    face_detection_openvino[4].start_async(request_id=config.cur_request_id,
                                           inputs={face_detection_openvino[5]: face_in_frame})
    if face_detection_openvino[4].requests[config.cur_request_id].wait(-1) == 0:
        face_detection_res = face_detection_openvino[4].requests[config.cur_request_id].outputs[
            face_detection_openvino[6]]
        for face_loc in face_detection_res[0][0]:
            if face_loc[2] > 0.4:
                face_xmin = abs(int(face_loc[3] * width))
                face_ymin = abs(int(face_loc[4] * height))
                face_xmax = abs(int(face_loc[5] * width))
                face_ymax = abs(int(face_loc[6] * height))
                bboxes.append([face_xmin, face_ymin, face_xmax, face_ymax])
    return bboxes


def get_matches(face_reid_model, face, ret_encoded_face=False):
    encoded_face = encode(face_reid_model, face)
    script_query = get_script_query(encoded_face)
    search_response = es_search(es_client, script_query)
    result = Counter([hit['_source']['id'] for hit in search_response['hits']['hits']
                      if hit['_score'] > 0.60])
    if ret_encoded_face:
        return encoded_face, result
    else:
        return result


def init_streams():
    _out_streams = []
    _in_stream = WebcamVideoStream(src=config.in_stream).start()
    for stream in config.out_stream:
        _out_streams.append(WebcamVideoStream(src=stream).start())
    return _in_stream, _out_streams


def main():
    next_id = 0
    while True:
        in_stream_frame = in_stream.read()
        out_stream_frames = []
        for stream in out_streams:
            out_stream_frames.append(stream.read())

        bboxes = face_detection(in_stream_frame, face_detection_openvino)
        for bbox in bboxes:
            encoded_face, result = get_matches(face_reid_openvino, in_stream_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], True)
            if len(result) == 0:
                bulk(es_client, [{"_op_type": "index", "_index": config.es_index_name,
                                  "id": str(next_id), "face_vector": encoded_face[0].tolist()}])
                next_id += 1
            else:
                most_common_id = result.most_common(1)[0]
                if most_common_id[1] < config.es_max_features_size:
                    bulk(es_client, [{"_op_type": "index", "_index": config.es_index_name,
                                      "id": str(most_common_id[0]), "face_vector": encoded_face[0].tolist()}])
                else:
                    pass

        for out_frame in out_stream_frames:
            bboxes = face_detection(out_frame, face_detection_openvino)
            for bbox in bboxes:
                result = get_matches(face_reid_openvino, out_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                if len(result) == 0:
                    user = 'unknown'
                else:
                    most_common_id = result.most_common(1)[0][0]
                    user = str(most_common_id)
                cv2.rectangle(out_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                cv2.putText(out_frame, user, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        cv2.imshow('input', in_stream_frame)
        cv2.imshow('output', out_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == '__main__':
    in_stream, out_streams = init_streams()
    face_detection_openvino = load_xml_bin('model/face-detection-adas-0001.xml')
    face_reid_openvino = load_xml_bin('model/face-reidentification-retail-0095.xml')
    es_client = init_es()
    if es_client is None:
        print('[WARNING] elasticsearch is offline')
        exit()
    main()
