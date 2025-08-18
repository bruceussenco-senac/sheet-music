import cv2
import numpy as np

n_count = 3

# Load the main image and the template image
main_color = cv2.imread('ode_to_joy.png', cv2.IMREAD_COLOR)
main_image = cv2.cvtColor(main_color, cv2.COLOR_BGR2GRAY)
templates = []
for i in range(n_count):
    template = cv2.imread(f'n{i}.png', cv2.IMREAD_COLOR)
    templates.append(template)

threshold = 0.8

for template in templates:
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(main_image, template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    # Draw rectangles around detected areas
    for pt in zip(*locations[::-1]):
        cv2.rectangle(main_color, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)

# Display the result
detected_image = cv2.resize(main_color, (int(main_color.shape[1]/2), int(main_color.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.imshow('Detected Image', detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
