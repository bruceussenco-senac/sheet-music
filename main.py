import cv2
import numpy as np

threshold = 0.8

def clef_read():
    template_gray = cv2.cvtColor(clef_template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(main_image, template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    for pt in zip(*locations[::-1]):
        cv2.rectangle(main_color, pt, (pt[0] + clef_template.shape[1], pt[1] + clef_template.shape[0]), (0, 0, 255), 2)
        clefs_heights.append(int(pt[1]))



n_count = 4

clefs_heights = []

# Load the main image and the template image
main_color = cv2.imread('ode_to_joy.png', cv2.IMREAD_COLOR)
main_image = cv2.cvtColor(main_color, cv2.COLOR_BGR2GRAY)
note_templates = []
clef_template = cv2.imread('clef0.png', cv2.IMREAD_COLOR)
for i in range(n_count):
    template = cv2.imread(f'n{i}.png', cv2.IMREAD_COLOR)
    note_templates.append(template)

note_positions = []
note_heights = set()

clef_read()

for template in note_templates:
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(main_image, template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    # Draw rectangles around detected areas
    for pt in zip(*locations[::-1]):
        cv2.rectangle(main_color, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)
        note_positions.append((int(pt[0]), int(pt[1])))
        note_heights.add(pt[1])

# Draw lines for heights
for h in note_heights:
    cv2.line(main_color, (0, h), (main_color.shape[0], h), (255, 0, 0))

# Display the result
detected_image = cv2.resize(main_color, (int(main_color.shape[1]/2), int(main_color.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.imshow('Detected Image', detected_image)

for pos in note_positions:
    print(pos)

note_heights = sorted(note_heights)
for h in note_heights:
    print(h)

cv2.waitKey(0)
cv2.destroyAllWindows()
