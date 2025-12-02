import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment


def extract_shapes(image_path):
    # Load and convert to binary
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

    # Find connected components
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        binary, connectivity=8
    )

    shapes = []
    for i in range(1, num_labels):  # Ignore the background (label 0)
        # Create a mask for this component
        mask = (labels == i).astype(np.uint8) * 255

        # Find the contour
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours and len(contours[0]) >= 5:  # At least 5 points to be a valid shape
            shapes.append(
                {
                    "contour": contours[0],
                    "area": stats[i, cv.CC_STAT_AREA],
                    "centroid": centroids[i],
                }
            )

    return shapes


def compare_two_shapes(contour1, contour2):
    # matchShapes returns a distance (0 = identical, larger means more different)
    distance = cv.matchShapes(contour1, contour2, cv.CONTOURS_MATCH_I2, 0)

    # Convert to similarity score 0-100
    # Typical distance: 0-0.5 for similar shapes, >1 for very different
    similarity = max(0, 100 - (distance * 100))

    return similarity


def match_shapes_between_images(shapes1, shapes2):
    if not shapes1 or not shapes2:
        return [], 0

    n1, n2 = len(shapes1), len(shapes2)

    # Cost matrix (distance between each pair of shapes)
    cost_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            similarity = compare_two_shapes(
                shapes1[i]["contour"], shapes2[j]["contour"]
            )
            cost_matrix[i, j] = 100 - similarity  # Cost = inverse of similarity

    # Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for i, j in zip(row_ind, col_ind):
        similarity = 100 - cost_matrix[i, j]
        matches.append({"model_idx": i, "drawing_idx": j, "similarity": similarity})

    return matches, len(shapes1)


def evaluate_drawing(template: str, image: str, min_similarity=60):
    # Extract shapes from both images
    model_shapes = extract_shapes(template)
    drawing_shapes = extract_shapes(image)

    # Check that shapes were found
    if not model_shapes:
        return {
            "score": 0,
            "quality": "Error",
            "details": "No shapes detected in the template",
        }

    if not drawing_shapes:
        return {
            "score": 0,
            "quality": "Very Bad",
            "details": "No shapes detected in the drawing",
        }

    # Associate shapes
    matches, expected_count = match_shapes_between_images(model_shapes, drawing_shapes)

    # Calculate scores
    num_model = len(model_shapes)
    num_drawing = len(drawing_shapes)

    # Penalty if incorrect number of shapes
    count_penalty = abs(num_model - num_drawing) * 15

    # Similarity score of matched shapes
    if matches:
        shape_scores = [m["similarity"] for m in matches]
        avg_similarity = np.mean(shape_scores)

        # Count how many shapes are "correct"
        correct_shapes = sum(1 for s in shape_scores if s >= min_similarity)

        # Final score
        # - 70% for average similarity of shapes
        # - 30% for number of correct shapes
        final_score = (
            avg_similarity * 0.7
            + (correct_shapes / expected_count * 100) * 0.3
            - count_penalty
        )
    else:
        avg_similarity = 0
        correct_shapes = 0
        final_score = 0

    final_score = max(0, min(100, final_score))

    # Determine quality
    if final_score >= 85:
        quality = "Excellent"
    elif final_score >= 70:
        quality = "Good"
    elif final_score >= 50:
        quality = "Average"
    else:
        quality = "Bad"

    return {
        "score": round(final_score, 1),
        "quality": quality,
        "avg_similarity": round(avg_similarity, 1),
        "shapes_expected": num_model,
        "shapes_found": num_drawing,
        "shapes_correct": correct_shapes if matches else 0,
        "details": f"{correct_shapes}/{expected_count} formes correctes, similarité moyenne: {avg_similarity:.1f}%",
    }


if __name__ == "__main__":
    template = "assets/template.png"

    images = [
        "assets/bigger.png",
        "assets/smaller.png",
        "assets/wrong.png",
    ]

    for image in images:
        print("\n" + "=" * 60)

        result = evaluate_drawing(template, image)

        print(f"\n{image}:")
        print(f"  Overall Score: {result['score']}% ({result['quality']})")
        print(f"  Average Similarity: {result['avg_similarity']}%")
        print(
            f"  Shapes: {result['shapes_correct']}/{result['shapes_expected']} correct"
        )
        print(f"  Detected: {result['shapes_found']} shapes")
        print(f"  Details: {result['details']}")

        print("\n" + "=" * 60)
