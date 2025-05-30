import cv2
import numpy as np

def detect_ball_by_contour(frame, min_radius=40):
    """
    Detects large round objects using contour and enclosing circle.
    Draws circles and bounding boxes around detected balls.

    Parameters:
        frame (np.ndarray): BGR input image
        min_radius (int): Minimum radius of detected circle to ignore small noise

    Returns:
        output (np.ndarray): Image with annotations
    """
    output = frame.copy()
    
    # Preprocess image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Edge detection
    edged = cv2.Canny(blurred, 30, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > min_radius:
            if len(contour) >= 20 and len(contour) <= 80:  
                # Draw the enclosing circle
                cv2.circle(output, (int(x), int(y)), int(radius), (0, 255, 0), 2)
    return output


if __name__ == "__main__":
    # Load image
    img = cv2.imread("../../assets/imgs/ball.jpg")  # Replace with your image file path

    # Optionally resize if it's too large
    img = cv2.resize(img, (800, 600))  # Resize for faster processing/display

    result = detect_ball_by_contour(img)

    # Show result
    cv2.imshow("Detected Ball (Contour Method)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()