import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def crop_image(image):
    coordinates = np.column_stack(np.where(image == 0))

    top_left = coordinates.min(axis=0)
    bottom_right = coordinates.max(axis=0)

    top_left = np.maximum(top_left - 5, 0)
    bottom_right = np.minimum(bottom_right + 6, np.array(image.shape) - 1)

    cropped_image = image[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]

    return cropped_image


def process_image(file_path, threshold_value):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    _, threshold_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    cropped_image = crop_image(threshold_image)

    return cropped_image


def display_image(image, segments, name_image):
    height, width = image.shape
    num_segments = len(segments)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    for i in range(num_segments):
        x = i * (width // num_segments)
        ax.axvline(x=x, color='g', linestyle='-', linewidth=1)
        ax.text(x + 2, height // 2, str(i + 1), color='g', fontsize=10, verticalalignment='center')

    ax.axvline(x=num_segments * (width//num_segments), color='g', linestyle='-', linewidth=1)

    plt.title(name_image)
    plt.show()


def calculate_absolute_vector(image, num_stripes):
    height, width = image.shape
    stripe_width = width // num_stripes
    feature_vector = []
    for i in range(num_stripes):
        segment = image[:, i * stripe_width:(i + 1) * stripe_width]
        black_pixels = np.sum(segment == 0)
        feature_vector.append(black_pixels)

    return feature_vector


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

    image_files = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print(f"Обробка зображення: {image_file}")

        image = process_image(image_path, threshold)

        abs_vector = calculate_absolute_vector(image, num_stripes=4)
        YenakiiS1 = normalize_sum(abs_vector)
        YenakiiM1 = normalize_max(abs_vector)

        print(f"Абсолютний вектор: {abs_vector}")
        print(f"Нормований вектор (за сумою): {YenakiiS1}")
        print(f"Нормований вектор (за максимумом): {YenakiiM1}")

        display_image(image, abs_vector, image_file)


if __name__ == "__main__":
    main()
