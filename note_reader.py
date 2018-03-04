import cv2
import numpy as np
import pandas as pd

def find_notes(filepath):
    lines = find_lines(filepath)
    filtered_image = apply_watershed_filter(filepath)
    blobs = find_blobs_from_watershed(filepath, filtered_image, )
    notes = map_blob_center_to_lines(blobs, lines)

    return notes

def find_lines(filepath):
    img = cv2.imread(filepath)
    img = img[:, int(img.shape[1]*.05):]
    img = cv2.resize(img, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200,apertureSize = 3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 4500)

    TILT_TOLERANCE_DEGREES = 1

    UP_TILT_LIMIT =  ((90 - TILT_TOLERANCE_DEGREES) * np.pi) /180
    DOWN_TILT_LIMIT = ((90 + TILT_TOLERANCE_DEGREES) * np.pi) /180

    LINE_LENGTH = 6200
    LINE_THICKNESS = 1
    LINE_COLOR = (255,0,255)

    output_filepath = "lines_detected.png"

    hline_data = []
    y_intercepts = []
    for line in lines:
        for rho,theta in line:
            is_horizontal = (theta < DOWN_TILT_LIMIT) and (theta > UP_TILT_LIMIT)
            if is_horizontal:
                y_intercepts.append(rho)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + LINE_LENGTH*(-b))
                y1 = int(y0 + LINE_LENGTH*(a))
                x2 = int(x0 - LINE_LENGTH*(-b))
                y2 = int(y0 - LINE_LENGTH*(a))
                if is_horizontal:
                    hline_data.append((y1 + y2) / 2)
                if output_filepath:
                    cv2.line(img,(x1,y1),(x2,y2), LINE_COLOR, LINE_THICKNESS)
                    cv2.imwrite(output_filepath,img)

    batched_lines = []

    GAP_THRESHOLD_PERCENT = 0.03

    y_intercepts.sort()
    line_df = pd.DataFrame(
        {'id' : pd.Series(np.zeros(len(y_intercepts))),
         'intercept' : pd.Series(y_intercepts),
         'gap_to_next' : pd.Series(y_intercepts).diff(-1).abs()
         }
    )
    line_id = 0
    for i in line_df.index:
        line_df.loc[i,'id'] = line_id
        if line_df.ix[i, 'gap_to_next'] > (GAP_THRESHOLD_PERCENT*img.shape[0]):
            line_id += 1
    batched = line_df[['id','intercept']].groupby('id').mean()

    print "Detected %d horizontal lines" % (
        len(batched))

    assert len(batched) == 5, "Found incorrect number of staff lines"

    return batched

def apply_watershed_filter(filepath):
    img = cv2.imread(filepath)
    img = img[:, int(img.shape[1]*.05):]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    cv2.imwrite("circle_experiment.png", unknown)
    
    return unknown

def find_blobs_from_watershed(filepath, watershed_image):
    img = cv2.imread(filepath)
    img = img[:, int(img.shape[1]*.05):]
    detector = cv2.SimpleBlobDetector_create()

    blobs = detector.detect(watershed_image)

    img_with_blobs = cv2.drawKeypoints(
        img, blobs,
        np.array([]),
        (0,0,255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite("circle_experiment.png", img_with_blobs)

    return blobs

def map_blob_center_to_lines(blobs, lines):
    x_pos = [b.pt[0]*4 for b in blobs]
    y_pos = [b.pt[1]*4 for b in blobs]
    blob_positions = pd.DataFrame({
        'x' : x_pos,
        'y' : y_pos
    })
    blob_positions.sort_values('x', inplace=True)
    cutoffs = create_cutoffs(lines)
    notes = []
    for y in blob_positions.y:
        notes.append(map_blob_to_notes(y, cutoffs))
    return notes

## Helper functions ##

def create_cutoffs(lines):
    lines['gap_to_next'] = lines.intercept.diff(-1).abs()
    cutoffs = []
    for i in lines.index:
        this_line = lines.loc[i,].intercept
        if i == 0:
            cutoffs.append(this_line - (lines.loc[i,].gap_to_next*1.75))
            cutoffs.append(this_line - (lines.loc[i,].gap_to_next*1.25))
            cutoffs.append(this_line - (lines.loc[i,].gap_to_next*0.75))
            cutoffs.append(this_line - (lines.loc[i,].gap_to_next*0.25))
        if i < max(lines.index):
            cutoffs.append(this_line + (lines.loc[i,].gap_to_next*0.25))
            cutoffs.append(this_line + (lines.loc[i,].gap_to_next*0.75))
        else:
            cutoffs.append(this_line + (lines.loc[i-1,].gap_to_next*0.25))
            cutoffs.append(this_line + (lines.loc[i-1,].gap_to_next*0.75))
            cutoffs.append(this_line + (lines.loc[i-1,].gap_to_next*1.25))
            cutoffs.append(this_line + (lines.loc[i-1,].gap_to_next*1.75))
            cutoffs.append(this_line + (lines.loc[i-1,].gap_to_next*2.25))
            cutoffs.append(this_line + (lines.loc[i-1,].gap_to_next*2.75))
            cutoffs.append(this_line + (lines.loc[i-1,].gap_to_next*2.25))
    return cutoffs

def map_blob_to_notes(y, cutoffs):
    ordered_notes = [
        'C', 'B', 'A', 'G', 'F', 'E', 'D', 
        'C', 'B', 'A', 'G', 'F', 'E', 'D', 
        'C', 'B', 'A', 'G', 'F'
    ]
    for c, n in zip(cutoffs, ordered_notes):
        if y < c:
            return n
        continue
