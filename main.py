import cv2
import numpy as np
import math

threshold = 0.8

def clef_read():
    template_gray = cv2.cvtColor(clef_template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(main_image, template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    clef_count = 0

    for pt in zip(*locations[::-1]):
        clefs_heights.append(int(pt[1]))
        clef_count += 1
    
    return clef_count

def note_read():
    for template in note_templates:
        tem_w = template.shape[1]
        tem_h = template.shape[0]
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(main_image, template_gray, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1]):
            note_positions.append((
                int(pt[0]) + int(tem_w/2),
                int(pt[1]) + int(tem_h/2)
            ))

def round_notes():
    for note_id in range(len(note_positions)):
        note_height = note_positions[note_id][1]
        closest_clef_id = get_closest_clef(note_height)

        dist = note_positions[note_id][1] - clefs_heights[closest_clef_id]
        tone_dist = round(dist/line_hheight)
        dist = math.floor(tone_dist * line_hheight)

        note_positions[note_id] = (note_positions[note_id][0], clefs_heights[closest_clef_id] + dist)

        notes_by_clefs[closest_clef_id].append((note_positions[note_id][0], tone_dist))

def get_closest_clef(note_height):
    closest_clef_id = 0
    dist = 10000000000
    for i in range(len(clefs_heights)):
        clef_top = clefs_heights[i]
        clef_bottom = clefs_heights[i] + clef_template_height
        new_dist = min(abs(clef_top - note_height), abs(clef_bottom - note_height))

        if new_dist < dist:
            dist = new_dist
            closest_clef_id = i

    return closest_clef_id

def order_notes_in_clef():
    for i in range(len(notes_by_clefs)):
        notes_by_clefs[i] = sorted(notes_by_clefs[i], key=lambda x: x[0])


def draw():
    # Line references with clefs positions
    for h in clefs_heights:
        ch = h + clef_template.shape[0] - 1
        cv2.line(main_color, (500, h), (main_color.shape[0], h), (0, 0, 255))
        cv2.line(main_color, (500, ch), (main_color.shape[0], ch), (0, 0, 255))

        for i in range(1, 8):
            cv2.line(main_color, (500, h+int(i*line_hheight)), (main_color.shape[0], h+int(i*line_hheight)), (0, 127, 255))

    # draw a line from the to note to its clef-figure
    for clef_index, clef_notes in enumerate(notes_by_clefs):
        for note in clef_notes:
            note_x = note[0]
            note_y = int(clefs_heights[clef_index] + math.floor(note[1] * line_hheight))
            cv2.line(
                main_color,
                (int(note_x), note_y),
                (int(note_x), int(clefs_heights[clef_index])),
                (127, 127, 0)
            )
            cv2.rectangle(main_color, (note_x, note_y), (note_x, note_y), (255, 0, 0))


n_count = 4 # note template count

# Load the main image and the template image
main_color = cv2.imread('ode_to_joy.png', cv2.IMREAD_COLOR)
main_image = cv2.cvtColor(main_color, cv2.COLOR_BGR2GRAY)
note_templates = []
clef_template = cv2.imread('clef0.png', cv2.IMREAD_COLOR)
clef_template_height = clef_template.shape[0]
line_height = (clef_template.shape[0]-1) / 4 # distance between consecutive lines of the sheet
line_hheight = (clef_template.shape[0]-1) / 8 # "distance between notes", "tone" distance
for i in range(n_count):
    template = cv2.imread(f'n{i}.png', cv2.IMREAD_COLOR)
    note_templates.append(template)

note_positions = []
clefs_heights = []

clef_count = clef_read()
notes_by_clefs = [[] for _ in range(clef_count)] # array with a array for each clef with its notes
                                                 # notes are represented with (x position, distance from the top of the clefs in "tones")
note_read()
round_notes()
order_notes_in_clef()

for clef_notes in notes_by_clefs:
    print()
    print(clef_notes)

# Display the result
draw()
detected_image = cv2.resize(main_color, (int(main_color.shape[1]/2), int(main_color.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.imshow('Detected Image', detected_image)
cv2.imwrite('export.png', main_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
