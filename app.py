from flask import Flask, request, jsonify, send_file
import os
import cv2
import numpy as np
import glob
import tempfile
import time
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from model_detection import setup_cfg, get_parser, VisualizationDemo, save_result_to_txt

# Flask 애플리케이션 생성
app = Flask(__name__)

# Detectron2 설정 및 모델 불러오기
cfg = get_cfg()
cfg.merge_from_file("./config.yaml")  # YAML 파일 경로 설정
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 추론 임계값 설정
cfg.MODEL.WEIGHTS = "./model_0000599.pth"  # 모델 가중치 파일 경로 설정
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU가 사용 가능하면 GPU 사용

predictor = DefaultPredictor(cfg)

@app.route('/detect_visual', methods=['POST'])
def detect_objects_visual():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        # 이미지를 읽어들이기
        file = request.files['image']
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # 이미지가 제대로 로드되지 않았을 경우 오류 응답
        if img is None:
            return jsonify({"error": "Failed to load image"}), 400

        # Detectron2를 사용한 예측 수행
        outputs = predictor(img)

        # 예측된 객체들 가져오기
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()

        # 잘라낸 이미지를 저장할 리스트
        cropped_images = []

        # 바운딩 박스를 기준으로 이미지 자르기
        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = box

            # 여유 공간 추가 (10픽셀씩)
            x1 = max(0, int(x1) - 10)
            y1 = max(0, int(y1) - 10)
            x2 = min(img.shape[1], int(x2) + 10)
            y2 = min(img.shape[0], int(y2) + 10)

            # 이미지 자르기
            cropped_img = img[y1:y2, x1:x2]
            cropped_images.append(cropped_img)

            # 잘라낸 이미지 파일로 저장 (옵션)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_class_{cls}.jpg')
            temp_filename = temp_file.name
            cv2.imwrite(temp_filename, cropped_img)

        # 결과 확인
        print(f"총 {len(cropped_images)}개의 객체가 잘려 저장되었습니다.")

        # 클라이언트에게 이미지 파일 전송
        return send_file(temp_filename, mimetype='image/jpeg')

    except Exception as e:
        # 예외 발생 시 오류 메시지 반환
        return jsonify({"error": str(e)}), 500

@app.route('/batch_process', methods=['POST'])
def batch_process():
    try:
        # 입력 경로와 출력 경로 가져오기
        input_dir = request.json.get('input_dir')
        output_dir = request.json.get('output_dir')

        if not os.path.exists(input_dir):
            return jsonify({"error": "Input directory does not exist"}), 400

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # VisualizationDemo 초기화
        args = get_parser().parse_args([])  # 빈 인자 전달로 argparse 초기화
        args.input = input_dir + '/*.jpg'
        args.output = output_dir
        cfg = setup_cfg(args)
        detection_demo = VisualizationDemo(cfg)

        # 배치 처리 수행
        start_time_all = time.time()
        img_count = 0

        for img_path in glob.glob(args.input):
            print(f"Processing {img_path}...")
            img_name = os.path.basename(img_path)
            img_save_path = os.path.join(args.output, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Failed to load {img_path}")
                continue

            start_time = time.time()

            prediction, vis_output, polygons = detection_demo.run_on_image(img)

            # 결과 저장
            txt_save_path = os.path.join(args.output, f"res_img_{img_name.split('.')[0]}.txt")
            save_result_to_txt(txt_save_path, prediction, polygons)
            vis_output.save(img_save_path)

            print(f"Time: {time.time() - start_time:.2f} s / img")
            img_count += 1

        avg_time = (time.time() - start_time_all) / img_count if img_count > 0 else 0
        print(f"Average Time: {avg_time:.2f} s / img")

        return jsonify({"message": "Batch processing completed", "average_time": avg_time}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)