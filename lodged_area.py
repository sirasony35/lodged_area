import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def calculate_lodged_area_by_exg(image_path):
    """
    Excess Green (ExG) 식생 지수와 Otsu의 자동 임계값 설정을 사용하여
    도복된 벼의 면적을 계산하고 시각화하는 함수.
    """
    try:
        # 1. 이미지 불러오기
        pil_img = Image.open(image_path)
        image_rgb = np.array(pil_img)
        # ------------------- 수정된 부분 -------------------
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # 오타 수정: COLOR_RGB_BGR -> COLOR_RGB2BGR
        # ----------------------------------------------------
    except FileNotFoundError:
        print(f"오류: '{image_path}' 파일을 찾을 수 없습니다. 코드와 이미지가 같은 폴더에 있는지 확인하세요.")
        return -1
    except Exception as e:
        print(f"이미지를 읽는 중 오류 발생: {e}")
        return -1

    # 2. Excess Green (ExG) 지수 계산
    B, G, R = cv2.split(image_bgr)
    R = R.astype(np.int16)
    G = G.astype(np.int16)
    B = B.astype(np.int16)
    exg = 2 * G - R - B
    exg_normalized = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. 마스크 생성 (Otsu의 자동 임계값 설정)
    threshold_value, mask_lodged = cv2.threshold(exg_normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask_lodged = cv2.morphologyEx(mask_lodged, cv2.MORPH_OPEN, kernel)
    mask_lodged = cv2.morphologyEx(mask_lodged, cv2.MORPH_CLOSE, kernel)

    # 5. 면적 계산
    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    lodged_pixels = cv2.countNonZero(mask_lodged)
    lodged_area_ratio = lodged_pixels / total_pixels

    print(f"\n분석 완료!")
    print(f"자동으로 설정된 임계값: {threshold_value:.2f}")
    print(f"계산된 도복 면적 비율: {lodged_area_ratio * 100:.2f}%")

    # 6. 결과 시각화
    overlay = image_bgr.copy()
    overlay[mask_lodged > 0] = (0, 0, 255)
    alpha = 0.4
    overlay_image = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(exg_normalized, cmap='gray')
    plt.title('Excess Green (ExG) Index')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    plt.title('Overlay Result (Lodged Area)')
    plt.axis('off')

    plt.suptitle(f"Estimated Lodged Area Ratio: {lodged_area_ratio * 100:.2f}%", fontsize=16)
    plt.tight_layout()
    plt.show()

    return lodged_area_ratio

# --- 실행 부분 ---
IMAGE_FILE_PATH = 'HP01_03_251017_RGB.tif'
calculate_lodged_area_by_exg(IMAGE_FILE_PATH)