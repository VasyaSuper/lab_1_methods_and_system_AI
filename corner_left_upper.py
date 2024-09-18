import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def crop_image(image):
    coordinates = np.column_stack(np.where(image == 0))

    if coordinates.size == 0:
        return image

    top_left = coordinates.min(axis=0)
    bottom_right = coordinates.max(axis=0)

    # top_left = np.maximum(top_left - 5, 0)
    # bottom_right = np.minimum(bottom_right + 6, np.array(image.shape) - 1)

    cropped_image = image[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]

    return cropped_image


def process_image(file_path, threshold_value):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    _, threshold_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    cropped_image = crop_image(threshold_image)
    return cropped_image


def calculate_feature_vector(image, num_segments):
    height, width = image.shape
    feature_vector = []
    angles = np.linspace(0, 90, num_segments + 1)
    angles = angles[1:]

    # print(angles)

    for angle in angles:
        rad_angle = np.radians(angle)
        x_end = width
        y_end = int(x_end * np.tan(rad_angle))

        if y_end > height:
            y_end = height
            x_end = int(y_end / np.tan(rad_angle))

        mask = np.zeros((height, width), dtype=np.uint8)

        polygon_points = np.array([
            [0, 0],
            [x_end, y_end],
            [width, height],
            [0, height]
        ], np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))

        cv2.fillPoly(mask, [polygon_points], 255)

        # print('++++++++++++++++++++++++++++++++++++')
        # print(img_test)
        # print('------')
        # print(image)

        # fig, ax = plt.subplots()
        # ax.imshow(mask, cmap='gray')

        masked_image = np.zeros((height, width), dtype=np.uint8)
        for x in range(width):
            for y in range(height):
                if image[y, x] == mask[y, x] and image[y, x] == 0:
                    masked_image[y, x] = 1

        # print("-=-=-=-=-")
        # print(masked_image)

        black_pixels = np.sum(masked_image == 1)

        previous_sum = sum(feature_vector)
        adjusted_black_pixels = black_pixels - previous_sum
        feature_vector.append(adjusted_black_pixels)

    return feature_vector


def display_image(image, num_segments, name_image):
    height, width = image.shape
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    angles = np.linspace(0, 90, num_segments + 1)[:-1]

    for angle in angles:
        rad_angle = np.radians(angle)
        x_end = width
        y_end = int(x_end * np.tan(rad_angle))

        if y_end > height:
            y_end = height
            x_end = int(y_end / np.tan(rad_angle))

        if angle == 90:
            x_end = 0
            y_end = height

        ax.plot([0, x_end], [0, y_end], color='g', linestyle='-', linewidth=1)
        ax.text(x_end + 2, y_end + 2, str(int(angle)), color='g', fontsize=10, verticalalignment='center')

    ax.plot([0, 0], [0, height], color='g', linestyle='-', linewidth=1)

    plt.title(name_image)
    plt.show()


def normalize_sum(vector):
    total_sum = sum(vector)

    if total_sum == 0:
        return [0 for v in vector]

    return [v / total_sum for v in vector]


def normalize_max(vector):
    max_value = max(vector)

    if max_value == 0:
        return [0 for v in vector]

    return [v / max_value for v in vector]


def main():
    image_folder = "C:/University/VII_semester/Methods_and_systems_of_artificial_intelligence/lab_1/images/"
    threshold = 150
    num_segments = 5

    image_files = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print(f"Обробка зображення: {image_file}")

        image = process_image(image_path, threshold)

        feature_vector = calculate_feature_vector(image, num_segments)
        YenakiiS1 = normalize_sum(feature_vector)
        YenakiiM1 = normalize_max(feature_vector)
        print(f"Абсолютний вектор: {feature_vector}")

        print(f"Нормований вектор (за сумою): {YenakiiS1}")
        print(f"Нормований вектор (за максимумом): {YenakiiM1}")

        display_image(image, num_segments, image_file)


if __name__ == "__main__":
    main()
