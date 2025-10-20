import torch
import cv2
import numpy as np
import os
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.features import rasterize
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import segmentation_models_pytorch as smp
import warnings
from segmentation_models_pytorch.losses import DiceLoss

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# --- 1단계, 2단계, 3단계 함수는 이전과 동일합니다 (변경 없음) ---
def create_training_chips_batch(tif_dir, gpkg_dir, output_dir, chip_size=512):
    print("--- 1단계: 학습용 데이터 조각 일괄 생성을 시작합니다 ---")
    img_out_dir = os.path.join(output_dir, 'images')
    mask_out_dir = os.path.join(output_dir, 'masks')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    tif_files = sorted([f for f in os.listdir(tif_dir) if f.endswith('.tif')])
    if not tif_files:
        print(f"경고: '{tif_dir}' 폴더에서 학습할 TIF 파일을 찾을 수 없습니다.")
        return
    chip_count = 0
    for tif_file in tif_files:
        base_name = os.path.splitext(tif_file)[0]
        tif_path = os.path.join(tif_dir, tif_file)
        gpkg_path = os.path.join(gpkg_dir, base_name + '.gpkg')
        if not os.path.exists(gpkg_path):
            print(f"경고: {tif_path}에 해당하는 GPKG 파일({gpkg_path})이 없습니다. 건너뜁니다.")
            continue
        print(f"처리 중: {tif_file}")
        with rasterio.open(tif_path) as raster:
            gdf = gpd.read_file(gpkg_path).to_crs(raster.crs)
            mask_array = rasterize(
                shapes=gdf.geometry, out_shape=raster.shape, transform=raster.transform,
                fill=0, default_value=1, dtype=np.uint8
            )
            for j in range(0, raster.height, chip_size):
                for i in range(0, raster.width, chip_size):
                    window = rasterio.windows.Window(i, j, chip_size, chip_size)
                    img_chip = raster.read([1, 2, 3], window=window)
                    if img_chip.shape[1] != chip_size or img_chip.shape[2] != chip_size:
                        continue
                    mask_chip = mask_array[window.row_off:window.row_off + chip_size,
                                window.col_off:window.col_off + chip_size]
                    if np.sum(mask_chip) > (chip_size * chip_size * 0.01) and img_chip.max() > 0:
                        img_chip_rgb = np.transpose(img_chip, (1, 2, 0))
                        chip_filename = f"chip_{chip_count}.png"
                        Image.fromarray(img_chip_rgb).save(os.path.join(img_out_dir, chip_filename))
                        Image.fromarray(mask_chip * 255).save(os.path.join(mask_out_dir, chip_filename))
                        chip_count += 1
    print(f"총 {len(tif_files)}개의 TIF 파일로부터 {chip_count}개의 학습용 데이터 조각을 생성했습니다.")


class LodgedRiceDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir, self.mask_dir = image_dir, mask_dir
        self.images = os.listdir(image_dir)
        if not self.images:
            raise ValueError("학습할 이미지가 없습니다. 1단계에서 데이터 조각이 올바르게 생성되었는지 확인하세요.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB")) / 255.0
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0
        image = image.transpose(2, 0, 1).astype(np.float32)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(mask)


def train_lodged_rice_model(data_dir, model_save_path):
    print("\n--- 2단계: AI 모델 학습을 시작합니다 ---")
    dataset = LodgedRiceDataset(os.path.join(data_dir, 'images'), os.path.join(data_dir, 'masks'))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation='sigmoid')
    loss_fn = DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습에 사용될 장치: {device}")
    model.to(device)
    epochs = 40
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(), model_save_path)
    print(f"모델 학습 완료 및 저장! ({model_save_path})")
    return model


# --- 4단계: ✨ 이미지 저장 라이브러리를 Pillow로 교체한 최종 예측 함수 ---
def predict_batch(model, predict_dir, output_dir, chip_size=512):
    print("\n--- 3단계: 새로운 필지들에 대한 일괄 예측을 시작합니다 ---")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

            # --- ✨ 수정된 이미지 처리 및 저장 로직 (Pillow 사용) ---
            original_img_data = np.transpose(raster.read([1, 2, 3]), (1, 2, 0))
            if original_img_data.dtype == 'uint16':
                original_img_uint8 = (original_img_data / 256).astype(np.uint8)
            else:
                original_img_uint8 = original_img_data.astype(np.uint8)

            # 1. 원본 이미지를 Pillow Image 객체로 변환
            base_image = Image.fromarray(original_img_uint8).convert("RGBA")

            # 2. 마스크를 씌울 오버레이 이미지를 Pillow로 생성 (빨간색, 40% 투명도)
            overlay = Image.new("RGBA", base_image.size)
            overlay_draw = ImageDraw.Draw(overlay)
            # 픽셀 단위로 직접 접근하여 마스크 그리기
            pixels = overlay.load()
            for y in range(base_image.height):
                for x in range(base_image.width):
                    if prediction_full[y, x] == 1:
                        pixels[x, y] = (255, 0, 0, int(255 * 0.4))  # R, G, B, Alpha (투명도)

            # 3. 원본과 오버레이를 알파 채널을 이용해 합성
            final_image = Image.alpha_composite(base_image, overlay)

            # 4. 이미지에 텍스트 추가
            draw = ImageDraw.Draw(final_image)
            try:
                # 폰트가 시스템에 있는 경우 사용
                font = ImageFont.truetype("malgun.ttf", 40)
            except IOError:
                # 없는 경우 기본 폰트 사용
                font = ImageFont.load_default()

            text1 = f"File: {tif_file}"
            text2 = f"Lodged Ratio: {ratio * 100:.2f}%"
            text3 = f"Lodged Area: {lodged_area_m2:.2f} m2 ({lodged_area_pyeong:.2f} pyeong)"

            draw.rectangle([(10, 10), (900, 160)], fill=(0, 0, 0, 180))  # 반투명 검정 배경
            draw.text((20, 20), text1, font=font, fill=(255, 255, 255, 255))
            draw.text((20, 65), text2, font=font, fill=(255, 255, 255, 255))
            draw.text((20, 110), text3, font=font, fill=(255, 255, 255, 255))

            # 5. Pillow로 최종 이미지 저장
            try:
                print(f"결과 이미지 저장 시도: {output_vis_path}")
                final_image.save(output_vis_path, "PNG")
                print("-> 이미지 저장 성공!")
            except Exception as e:
                print(f"-> !!! 이미지 저장 실패 !!! 원인: {e}")
            # --- ✨ 여기까지 수정 ---

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


# --- 메인 실행 흐름 (변경 없음) ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(script_dir, 'data')
    TRAIN_TIF_DIR = os.path.join(DATA_DIR, 'train_tifs')
    TRAIN_GPKG_DIR = os.path.join(DATA_DIR, 'train_gpkg')
    PREDICT_TIF_DIR = os.path.join(DATA_DIR, 'predict_tifs')
    TRAINING_CHIPS_DIR = os.path.join(script_dir, 'training_chips')
    PREDICTIONS_DIR = os.path.join(script_dir, 'predictions')
    MODEL_PATH = os.path.join(script_dir, 'lodged_rice_model_final.pth')

    if not os.path.exists(TRAIN_TIF_DIR) or not os.path.exists(TRAIN_GPKG_DIR):
        print(f"오류: 학습 데이터 폴더를 찾을 수 없습니다!")
        print(f"'{TRAIN_TIF_DIR}' 와 '{TRAIN_GPKG_DIR}' 경로를 확인해주세요.")
        exit()

    create_training_chips_batch(TRAIN_TIF_DIR, TRAIN_GPKG_DIR, TRAINING_CHIPS_DIR)
    trained_model = train_lodged_rice_model(TRAINING_CHIPS_DIR, MODEL_PATH)
    predict_batch(trained_model, PREDICT_TIF_DIR, PREDICTIONS_DIR)