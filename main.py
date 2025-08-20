import cv2
import numpy as np

threshold = 0.8

def clef_read():
    template_gray = cv2.cvtColor(clef_template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(main_image, template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    for pt in zip(*locations[::-1]):
        cv2.rectangle(main_color, pt, (pt[0] + clef_template.shape[1], pt[1] + clef_template.shape[0]-1), (0, 0, 255))
        clefs_heights.append(int(pt[1]))

def note_read():
    for template in note_templates:
        tem_w = template.shape[1]
        tem_h = template.shape[0]
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(main_image, template_gray, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1]):
            # Draw rectangles around detected areas
            cv2.rectangle(main_color, pt, (pt[0] + tem_w, pt[1] + tem_h), (0, 255, 0))
            note_positions.append((
                int(pt[0]) + int(tem_w/2),
                int(pt[1]) + int(tem_h/2)
            ))
            note_heights.add(pt[1] + int(tem_h/2))

n_count = 4 # note template count

# Load the main image and the template image
main_color = cv2.imread('ode_to_joy.png', cv2.IMREAD_COLOR)
main_image = cv2.cvtColor(main_color, cv2.COLOR_BGR2GRAY)
note_templates = []
clef_template = cv2.imread('clef0.png', cv2.IMREAD_COLOR)
line_height = (clef_template.shape[0]-1) / 4 # distance between consecutive lines of the sheet
line_hheight = (clef_template.shape[0]-1) / 8 # "distance between notes"
for i in range(n_count):
    template = cv2.imread(f'n{i}.png', cv2.IMREAD_COLOR)
    note_templates.append(template)

note_positions = []
note_heights = set()
clefs_heights = []

clef_read()
note_read()

# Line references with clefs positions
for h in clefs_heights:
    ch = h + clef_template.shape[0] - 1
    cv2.line(main_color, (500, h), (main_color.shape[0], h), (0, 0, 255))
    cv2.line(main_color, (500, ch), (main_color.shape[0], ch), (0, 0, 255))

    for i in range(1, 8):
        cv2.line(main_color, (500, h+int(i*line_hheight)), (main_color.shape[0], h+int(i*line_hheight)), (0, 127, 255))

# Draw notes positions
for pos in note_positions:
    x = pos[0]; y = pos[1]
    cv2.rectangle(main_color, (x, y), (x, y), (255, 0, 0))

# Display the result
detected_image = cv2.resize(main_color, (int(main_color.shape[1]/2), int(main_color.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.imshow('Detected Image', detected_image)
cv2.imwrite('export.png', main_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
