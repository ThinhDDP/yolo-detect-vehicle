import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# O-QUICKHULL Core Framework
# ==========================================

def o_quickhull(a, b, Pab, case_j):
    if not Pab:
        return []
    c = max(Pab, key=lambda p: orthogonal_distance(p, a, b, case_j))
    Pac = get_points_on_right_of_orthogonal_line(Pab, a, c, case_j)
    Pcb = get_points_on_right_of_orthogonal_line(Pab, c, b, case_j)
    return o_quickhull(a, c, Pac, case_j) + [c] + o_quickhull(c, b, Pcb, case_j)

def compute_coch(points):
    if len(points) <= 4:
        return points

    q1 = min(points, key=lambda p: (p[0], -p[1]))    
    q1_p = min(points, key=lambda p: (-p[1], p[0]))  
    q2 = min(points, key=lambda p: (p[1], p[0]))     
    q2_p = min(points, key=lambda p: (p[0], p[1]))   
    q3 = min(points, key=lambda p: (-p[0], p[1]))    
    q3_p = min(points, key=lambda p: (p[1], -p[0]))  
    q4 = min(points, key=lambda p: (-p[1], -p[0]))   
    q4_p = min(points, key=lambda p: (-p[0], -p[1])) 

    if (q1 == q1_p and q2 == q2_p and q3 == q3_p and q4 == q4_p):
        return [q1, q2, q3, q4]

    extreme_bounds = [q1, q1_p, q2, q2_p, q3, q3_p, q4, q4_p]
    filtered_points = discard_internal_points(points, extreme_bounds)

    P_q1_q1p = get_points_on_right_of_orthogonal_line(filtered_points, q1, q1_p, case_j=1)
    P_q2_q2p = get_points_on_right_of_orthogonal_line(filtered_points, q2, q2_p, case_j=2)
    P_q3_q3p = get_points_on_right_of_orthogonal_line(filtered_points, q3, q3_p, case_j=3)
    P_q4_q4p = get_points_on_right_of_orthogonal_line(filtered_points, q4, q4_p, case_j=4)

    hull = []
    hull.append(q1)
    hull.extend(o_quickhull(q1, q1_p, P_q1_q1p, case_j=1))
    hull.extend([q1_p, q2])
    hull.extend(o_quickhull(q2, q2_p, P_q2_q2p, case_j=2))
    hull.extend([q2_p, q3])
    hull.extend(o_quickhull(q3, q3_p, P_q3_q3p, case_j=3))
    hull.extend([q3_p, q4])
    hull.extend(o_quickhull(q4, q4_p, P_q4_q4p, case_j=4))
    
    return hull

# --- Stubs for exact geometric math (requires specific paper parameters) ---
def orient(a, b, c):
    """
    Helper function to evaluate orientation.
    Returns < 0 if point c is on the right of directed line ab.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def orthogonal_distance(point, a, b, case_j=None):
    """
    Calculates a value proportional to the orthogonal distance from 'point' to line 'ab'.
    Using the absolute value of the orient determinant avoids square roots and division 
    while preserving the exact magnitude ranking needed to find the furthest point.
    """
    return abs(orient(a, b, point))

def get_points_on_right_of_orthogonal_line(points, a, b, case_j=None):
    """
    Filters points strictly on the right of the directed line ab.
    According to the determinant formula, orient(a, b, p) < 0 indicates the right side.
    """
    return [p for p in points if orient(a, b, p) < 0]

def discard_internal_points(points, extreme_points):
    """
    Drops points inside the initial bounding polygon (e.g., the 8-point polygon)
    using the standard Ray-Casting Point-in-Polygon algorithm. 
    """
    def is_inside(p, poly):
        x, y = p
        inside = False
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            xi, yi = poly[i]
            xj, yj = poly[j]
            
            # A horizontal ray extending rightward crosses the edge if y is between yi and yj,
            # and the intersection x-coordinate is strictly greater than the point's x.
            # Python's short-circuit evaluation prevents division by zero since 
            # (yj - yi) is never 0 if the first condition is True.
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        return inside

    # Keep only points that evaluate to False for is_inside
    return [p for p in points if not is_inside(p, extreme_points)]

# ==========================================
# YOLO Integration Pipeline
# ==========================================

VEHICLE_CLASSES = [2, 3, 5, 7]

# Load pretrained yolo model
model = YOLO("yolo26s-seg.pt")

# Run inference
results = model.predict(source="traffic.jpg", classes=VEHICLE_CLASSES, line_width=1)

# Extract original image for custom OpenCV drawing
img = results[0].orig_img.copy()

# Check if any masks were detected
if results[0].masks is not None:
    # Iterate through the (x, y) coordinates of each detected mask
    for mask_points in results[0].masks.xy:
        
        # Convert YOLO float coordinates to integer tuples for the algorithm
        points = [(int(p[0]), int(p[1])) for p in mask_points]
        
        # Compute the Connected Orthogonal Convex Hull
        coch_polygon = compute_coch(points)
        
        # Draw the orthogonal hull
        if coch_polygon:
            # Reshape into the format cv2.polylines expects
            pts_array = np.array(coch_polygon, np.int32).reshape((-1, 1, 2))
            
            # Draw a green orthogonal convex hull (thickness=2)
            cv2.polylines(img, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)

# Display the custom drawn result rather than the default results[0].show()
cv2.imshow("O-QUICKHULL Overlaid Vehicles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()