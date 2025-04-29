from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import os


def resize_images_in_directory(directory, new_width, new_height):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    resize_img = img.resize((new_width, new_height), resample=Image.LANCZOS)
                resize_img.save(file_path)



def adjust_exposure(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_saturation(image, factor):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def adjust_shadows(image, factor):
    # This is a bit tricky and usually requires more advanced techniques.
    # For simplicity, we'll use a combination of brightness and contrast.
    image = ImageEnhance.Brightness(image).enhance(factor * 0.5 + 1)  # Adjust shadows as brightness
    image = ImageEnhance.Contrast(image).enhance(factor * 0.5 + 1)  # Adjust shadows impact as contrast
    return image


def adjust_hue(image, factor):
    image = image.convert("HSV")
    h, s, v = image.split()
    h = ImageOps.autocontrast(h, cutoff=0)  # Normalize histogram
    h = h.point(lambda p: p + int(255 * factor % 256))  # Adjust hue
    h = h.convert("L")
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def adjust_temperature(image, kelvin):
    # Define the blackbody xy chromaticity coordinates for a range of temperatures
    # (These are approximate values)
    blackbody_temps = {
        1000: (0.6499, 0.3498), 2000: (0.5772, 0.3971), 3000: (0.4974, 0.4408),
        4000: (0.4361, 0.4774), 5000: (0.3850, 0.5173), 6000: (0.3456, 0.5510),
        6500: (0.3323, 0.5649), 7000: (0.3210, 0.5762), 7500: (0.3111, 0.5854),
        8000: (0.3021, 0.5933), 8500: (0.2939, 0.6004), 9000: (0.2865, 0.6067),
        9500: (0.2800, 0.6124), 10000: (0.2740, 0.6174), 11000: (0.2647, 0.6258),
        12000: (0.2570, 0.6324), 13000: (0.2501, 0.6382), 14000: (0.2439, 0.6435),
        15000: (0.2384, 0.6482), 16000: (0.2334, 0.6525), 17000: (0.2289, 0.6564),
        18000: (0.2248, 0.6600), 19000: (0.2210, 0.6633), 20000: (0.2176, 0.6662)
    }

    closest_temp = min(blackbody_temps, key=lambda t: abs(t - kelvin))
    x, y = blackbody_temps[closest_temp]

    # Convert xy to RGB using the D65 white point (0.3127, 0.3290)
    def xy_to_rgb(x, y):
        r = 3.2406 * x - 1.5372 * y - 0.4986
        g = -0.9689 * x + 1.8758 * y + 0.0415
        b = 0.0557 * x - 0.2040 * y + 1.0570
        r, g, b = map(lambda c: min(max(c, 0), 1), [r, g, b])
        return int(r * 255), int(g * 255), int(b * 255)

    r, g, b = xy_to_rgb(x, y)

    # Create a color matrix to apply the white balance
    matrix = np.array([
        [r / 255, 0, 0],
        [0, g / 255, 0],
        [0, 0, b / 255]
    ], dtype=np.float32)

    # Convert image to NumPy array and apply the matrix
    image_np = np.array(image) / 255.0
    adjusted_image_np = np.dot(image_np[..., :3], matrix.T)
    adjusted_image_np = np.clip(adjusted_image_np, 0, 1) * 255
    adjusted_image = Image.fromarray(np.uint8(adjusted_image_np))

    return adjusted_image


def adjust_image(image, exposure_factor=1.0, saturation_factor=1.0, brightness_factor=1.0, contrast_factor=1.0,
                 shadow_factor=1.0, hue_factor=0, kelvin_temperature=6500):
    # Convert the skimage image to the PIL image
    pil_image = Image.fromarray(np.uint8(image * 255))
    # Adjust the exposure
    pil_image = adjust_exposure(pil_image, exposure_factor)
    # Adjust the saturation
    pil_image = adjust_saturation(pil_image, saturation_factor)
    # Adjust the brightness
    pil_image = adjust_brightness(pil_image, brightness_factor)
    # Adjust the contrast
    pil_image = adjust_contrast(pil_image, contrast_factor)
    # Adjust the shadow
    pil_image = adjust_shadows(pil_image, shadow_factor)
    # Adjust the hue
    pil_image = adjust_hue(pil_image, hue_factor)
    # Adjust the color temperature
    pil_image = adjust_temperature(pil_image, kelvin_temperature)
    # Convert the PIL image back to the numpy array and normalize it to the range of [0, 1]
    adjusted_image = np.array(pil_image) / 255.0

    return adjusted_image



