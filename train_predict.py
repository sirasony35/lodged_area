import torch
import cv2
import numpy as np
import os
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.features import rasterize, shapes
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import segmentation_models_pytorch as smp
import warnings
from segmentation_models_pytorch.losses import DiceLoss
from shapely.geometry import shape, Polygon, MultiPolygon

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# --- 1단계: 학습용 데이터 조각 일괄 생성 ---
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


# --- 2단계: 데이터셋 클래스 ---
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


# --- 3단계: AI 모델 학습 ---
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


# --- 4단계: 예측 마스크를 GPKG 벡터로 변환하는 함수 (새로 추가) ---
def mask_to_gpkg(mask_array, transform, crs, output_path, simplify_tolerance=1.0):
    """
    예측된 마스크(래스터)를 벡터 폴리곤으로 변환하여 GPKG 파일로 저장합니다.

    Args:
        mask_array: 예측 마스크 배열 (0과 1로 구성)
        transform: rasterio transform 객체
        crs: 좌표계 (CRS)
        output_path: 저장할 GPKG 파일 경로
        simplify_tolerance: 폴리곤 단순화 허용오차 (픽셀 단위, 클수록 단순화)
    """
    # 마스크에서 폴리곤 추출
    polygon_generator = shapes(mask_array.astype(np.int16), mask=(mask_array == 1), transform=transform)

    polygons = []
    for geom, value in polygon_generator:
        if value == 1:  # 도복 영역만 추출
            poly = shape(geom)
            # 폴리곤 단순화 (노이즈 제거)
            if simplify_tolerance > 0:
                poly = poly.simplify(simplify_tolerance, preserve_topology=True)

            # 유효한 폴리곤만 추가
            if poly.is_valid and not poly.is_empty:
                # 면적이 너무 작은 폴리곤 제거 (노이즈 제거)
                if poly.area > (simplify_tolerance * simplify_tolerance):
                    polygons.append(poly)

    if not polygons:
        print(f"  경고: 추출할 폴리곤이 없습니다. GPKG 파일을 생성하지 않습니다.")
        return False

    # GeoDataFrame 생성
    gdf = gpd.GeoDataFrame({
        'id': range(len(polygons)),
        'area_m2': [poly.area for poly in polygons],
        'geometry': polygons
    }, crs=crs)

    # GPKG 파일로 저장
    gdf.to_file(output_path, driver='GPKG')
    print(f"  -> GPKG 파일 저장 완료: {output_path} ({len(polygons)}개 폴리곤)")
    return True


# --- 5단계: 예측 및 GPKG 저장 (개선된 버전) ---
def predict_batch(model, predict_dir, output_dir, chip_size=512, save_gpkg=True, simplify_tolerance=1.0):
    """
    새로운 필지들에 대한 일괄 예측을 수행하고 결과를 저장합니다.

    Args:
        model: 학습된 모델
        predict_dir: 예측할 TIF 파일이 있는 디렉토리
        output_dir: 결과를 저장할 디렉토리
        chip_size: 조각 크기 (기본값: 512)
        save_gpkg: GPKG 파일 저장 여부 (기본값: True)
        simplify_tolerance: 폴리곤 단순화 허용오차 (기본값: 1.0)
    """
    print("\n--- 3단계: 새로운 필지들에 대한 일괄 예측을 시작합니다 ---")
    os.makedirs(output_dir, exist_ok=True)

    # GPKG 저장 폴더 생성
    if save_gpkg:
        gpkg_output_dir = os.path.join(output_dir, 'predicted_gpkg')
        os.makedirs(gpkg_output_dir, exist_ok=True)
        print(f"예측된 마스크를 GPKG 파일로 저장합니다: {gpkg_output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predict_files = sorted([f for f in os.listdir(predict_dir) if f.endswith('.tif')])
    if not predict_files:
        print(f"경고: '{predict_dir}' 폴더에서 예측할 TIF 파일을 찾을 수 없습니다.")
        return

    results = []
    for tif_file in predict_files:
        print(f"\n예측 중: {tif_file}")
        base_name = os.path.splitext(tif_file)[0]
        tif_path = os.path.join(predict_dir, tif_file)

        with rasterio.open(tif_path) as raster:
            pixel_width, pixel_height = raster.res
            pixel_area_m2 = pixel_width * pixel_height
            prediction_full = np.zeros((raster.height, raster.width), dtype=np.uint8)

            # 전체 이미지 예측
            for j in range(0, raster.height, chip_size):
                for i in range(0, raster.width, chip_size):
                    window = rasterio.windows.Window(i, j, chip_size, chip_size)
                    img_chip = raster.read([1, 2, 3], window=window)
                    if img_chip.max() == 0:
                        continue
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

            # 통계 계산
            lodged_pixels = np.sum(prediction_full)
            ratio = lodged_pixels / prediction_full.size
            lodged_area_m2 = lodged_pixels * pixel_area_m2
            lodged_area_pyeong = lodged_area_m2 / 3.3058

            # GPKG 파일 저장 (새로 추가)
            if save_gpkg:
                gpkg_output_path = os.path.join(gpkg_output_dir, base_name + '.gpkg')
                mask_to_gpkg(
                    prediction_full,
                    raster.transform,
                    raster.crs,
                    gpkg_output_path,
                    simplify_tolerance=simplify_tolerance
                )

            # 시각화 이미지 저장
            output_vis_path = os.path.join(output_dir, base_name + '_prediction.png')
            original_img_data = np.transpose(raster.read([1, 2, 3]), (1, 2, 0))
            if original_img_data.dtype == 'uint16':
                original_img_uint8 = (original_img_data / 256).astype(np.uint8)
            else:
                original_img_uint8 = original_img_data.astype(np.uint8)

            base_image = Image.fromarray(original_img_uint8).convert("RGBA")
            overlay = Image.new("RGBA", base_image.size)
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
                final_image.save(output_vis_path, "PNG")
                print(f"  -> 시각화 이미지 저장: {output_vis_path}")
            except Exception as e:
                print(f"  -> 이미지 저장 실패: {e}")

            results.append({
                'filename': tif_file,
                'lodged_area_ratio(%)': round(ratio * 100, 2),
                'lodged_area(m2)': round(lodged_area_m2, 2),
                'lodged_area(pyeong)': round(lodged_area_pyeong, 2)
            })

    # CSV 저장
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'prediction_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n모든 예측 완료! 결과가 '{output_dir}' 폴더에 저장되었습니다.")
    if save_gpkg:
        print(f"예측된 마스크가 '{gpkg_output_dir}' 폴더에 GPKG 파일로 저장되었습니다.")
        print("이 GPKG 파일들을 수정하여 새로운 학습 데이터로 사용할 수 있습니다!")
    print(df)


# --- 메인 실행 흐름 ---
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

    # 1단계: 학습 데이터 생성
    create_training_chips_batch(TRAIN_TIF_DIR, TRAIN_GPKG_DIR, TRAINING_CHIPS_DIR)

    # 2단계: 모델 학습
    trained_model = train_lodged_rice_model(TRAINING_CHIPS_DIR, MODEL_PATH)

    # 3단계: 예측 및 GPKG 저장
    predict_batch(
        trained_model,
        PREDICT_TIF_DIR,
        PREDICTIONS_DIR,
        save_gpkg=True,  # GPKG 저장 활성화
        simplify_tolerance=1.0  # 폴리곤 단순화 정도 (1.0 = 적당한 단순화)
    )
