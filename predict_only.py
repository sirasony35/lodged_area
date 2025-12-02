import torch
import cv2
import numpy as np
import os
import rasterio
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import segmentation_models_pytorch as smp
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# --- ✨ 이미지 저장 라이브러리를 Pillow로 교체한 최종 예측 함수 ---
def predict_batch(model, predict_dir, output_dir, chip_size=512):
    """
    학습된 모델을 사용하여 지정된 폴더의 모든 TIF 이미지에 대해 일괄 예측을 수행하고,
    비율, 면적(m²), 평수를 계산하여 결과물(이미지, CSV)을 저장합니다.
    """
    print("--- 새로운 필지에 대한 일괄 예측을 시작합니다 ---")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"예측에 사용될 장치: {device}")
    model.to(device)
    model.eval()

    predict_files = sorted([f for f in os.listdir(predict_dir) if f.endswith('.tif')])
    if not predict_files:
        print(f"경고: '{predict_dir}' 폴더에서 예측할 TIF 파일을 찾을 수 없습니다.")
        return

    results = []
    for tif_file in predict_files:
        print(f"예측 중: {tif_file}")
        with rasterio.open(os.path.join(predict_dir, tif_file)) as raster:
            pixel_width, pixel_height = raster.res
            pixel_area_m2 = pixel_width * pixel_height
            prediction_full = np.zeros((raster.height, raster.width), dtype=np.uint8)
            for j in range(0, raster.height, chip_size):
                for i in range(0, raster.width, chip_size):
                    window = rasterio.windows.Window(i, j, chip_size, chip_size)
                    img_chip = raster.read([1, 2, 3], window=window)
                    if img_chip.max() == 0: continue
                    h, w = img_chip.shape[1], img_chip.shape[2]
                    padded_chip = np.zeros((3, chip_size, chip_size), dtype=img_chip.dtype)
                    padded_chip[:, :h, :w] = img_chip
                    img_chip_norm = np.transpose(padded_chip, (1, 2, 0)) / 255.0
                    img_chip_tensor = torch.from_numpy(img_chip_norm.transpose(2, 0, 1).astype(np.float32)).unsqueeze(
                        0).to(device)
                    with torch.no_grad():
                        pred = model(img_chip_tensor)
                    pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                    prediction_full[j:j + h, i:i + w] = pred_mask[:h, :w]

            lodged_pixels = np.sum(prediction_full)
            ratio = lodged_pixels / prediction_full.size
            lodged_area_m2 = lodged_pixels * pixel_area_m2
            lodged_area_pyeong = lodged_area_m2 / 3.3058

            output_vis_path = os.path.join(output_dir, os.path.splitext(tif_file)[0] + '_prediction.png')

            original_img_data = np.transpose(raster.read([1, 2, 3]), (1, 2, 0))
            if original_img_data.dtype == 'uint16':
                original_img_uint8 = (original_img_data / 256).astype(np.uint8)
            else:
                original_img_uint8 = original_img_data.astype(np.uint8)

            base_image = Image.fromarray(original_img_uint8).convert("RGBA")
            overlay = Image.new("RGBA", base_image.size)
            overlay_draw = ImageDraw.Draw(overlay)
            pixels = overlay.load()
            for y in range(base_image.height):
                for x in range(base_image.width):
                    if prediction_full[y, x] == 1:
                        pixels[x, y] = (255, 0, 0, int(255 * 0.4))

            final_image = Image.alpha_composite(base_image, overlay)
            draw = ImageDraw.Draw(final_image)
            try:
                font = ImageFont.truetype("malgun.ttf", 40)
            except IOError:
                font = ImageFont.load_default()

            text1 = f"File: {tif_file}"
            text2 = f"Lodged Ratio: {ratio * 100:.2f}%"
            text3 = f"Lodged Area: {lodged_area_m2:.2f} m2 ({lodged_area_pyeong:.2f} pyeong)"

            draw.rectangle([(10, 10), (900, 160)], fill=(0, 0, 0, 180))
            draw.text((20, 20), text1, font=font, fill=(255, 255, 255, 255))
            draw.text((20, 65), text2, font=font, fill=(255, 255, 255, 255))
            draw.text((20, 110), text3, font=font, fill=(255, 255, 255, 255))

            try:
                print(f"결과 이미지 저장 시도: {output_vis_path}")
                final_image.save(output_vis_path, "PNG")
                print("-> 이미지 저장 성공!")
            except Exception as e:
                print(f"-> !!! 이미지 저장 실패 !!! 원인: {e}")

            results.append({
                'filename': tif_file,
                'lodged_area_ratio(%)': round(ratio * 100, 2),
                'lodged_area(m2)': round(lodged_area_m2, 2),
                'lodged_area(pyeong)': round(lodged_area_pyeong, 2)
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'prediction_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n모든 예측 완료! 결과가 '{output_dir}' 폴더에 저장되었습니다.")
    print(df)


# --- 메인 실행 흐름 (예측 전용) ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    PREDICT_TIF_DIR = os.path.join(script_dir, 'data', 'JC')
    MODEL_PATH = os.path.join(script_dir, 'lodged_rice_model_final.pth')
    PREDICTIONS_DIR = os.path.join(script_dir, 'predictions')

    if not os.path.exists(MODEL_PATH):
        print(f"오류: 학습된 모델 파일('{MODEL_PATH}')을 찾을 수 없습니다!")
        print("이전에 'train_predict.py'를 실행하여 모델을 먼저 학습시키고 저장했는지 확인해주세요.")
        exit()
    if not os.path.exists(PREDICT_TIF_DIR):
        print(f"오류: 예측할 이미지가 들어있는 폴더('{PREDICT_TIF_DIR}')를 찾을 수 없습니다!")
        exit()

    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1, activation='sigmoid')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"'{MODEL_PATH}'에서 학습된 모델을 성공적으로 불러왔습니다.")

    predict_batch(model, PREDICT_TIF_DIR, PREDICTIONS_DIR)