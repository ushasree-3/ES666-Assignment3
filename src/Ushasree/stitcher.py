import cv2
import numpy as np
import glob
import os

class PanaromaStitcher():
    def __init__(self):
        pass

    def compute_homography_matrix(self, source_points, destination_points):
        matrix_A = []
        for i in range(len(source_points)):
            x1, y1 = source_points[i]
            x2, y2 = destination_points[i]
            z1 = z2 = 1
            matrix_A.extend([
                [z2 * x1, z2 * y1, z2 * z1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2 * z1],
                [0, 0, 0, -z2 * x1, -z2 * y1, -z2 * z1, y2 * x1, y2 * y1, y2 * z1]
            ])

        matrix_A = np.array(matrix_A)
        _, _, V_transpose = np.linalg.svd(matrix_A)
        homography = V_transpose[-1].reshape((3, 3))

        return homography / homography[2, 2]  # Normalize

    def RANSAC_homography(self, source_pts, destination_pts, max_iterations=1000, distance_threshold=1):
        threshold = np.sqrt(5.99) * distance_threshold
        best_homography = None
        optimal_inliers = []
        max_inlier_count = 0

        for _ in range(max_iterations):
            indices = np.random.choice(source_pts.shape[0], 4, replace=False)
            sample_src = source_pts[indices]
            sample_dst = destination_pts[indices]

            H = self.compute_homography_matrix(sample_src, sample_dst)

            src_pts_homogeneous = np.hstack([source_pts, np.ones((source_pts.shape[0], 1))])  # Homogeneous coordinates
            projected_pts = np.dot(H, src_pts_homogeneous.T).T
            projected_pts /= projected_pts[:, 2].reshape(-1, 1)
            
            distances = np.linalg.norm(destination_pts - projected_pts[:, :2], axis=1)
            inlier_indices = np.where(distances < threshold)[0]
            
            if len(inlier_indices) > max_inlier_count:
                max_inlier_count = len(inlier_indices)
                best_homography = H
                optimal_inliers = inlier_indices
        
        if len(optimal_inliers) > 4:
            best_homography = self.compute_homography_matrix(source_pts[optimal_inliers], destination_pts[optimal_inliers])
        
        return best_homography, optimal_inliers

    def find_homography(self, img1, img2):
        sift_detector = cv2.SIFT_create()
        keypoints1, descriptors1 = sift_detector.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift_detector.detectAndCompute(img2, None)

        bf_matcher = cv2.BFMatcher()
        matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        print(f"Identified {len(good_matches)} valid matches between images.")
        
        if len(good_matches) < 4:
            print("Insufficient matches found.")
            return None

        src_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        homography, inliers = self.RANSAC_homography(src_points, dst_points, max_iterations=5000)
        return homography

    def apply_cylindrical_projection(self, image, focal_length):
        height, width = image.shape[:2]
        projected_image = np.zeros_like(image)
        center_x, center_y = width // 2, height // 2

        for x in range(width):
            for y in range(height):
                theta = (x - center_x) / focal_length
                h_ = (y - center_y) / focal_length
                X, Y, Z = np.sin(theta), h_, np.cos(theta)
                x_projected, y_projected = int(focal_length * X / Z + center_x), int(focal_length * Y / Z + center_y)

                if 0 <= x_projected < width and 0 <= y_projected < height:
                    projected_image[y, x] = image[y_projected, x_projected]
        
        return projected_image

    def merge_images(self, base_image, overlay_image):
        mask = (overlay_image > 0).astype(np.float32)
        merged_image = (mask * overlay_image + (1 - mask) * base_image).astype(base_image.dtype)
        return merged_image

    def crop_to_content(self, img):
        gray_image = np.mean(img, axis=2).astype(np.uint8)
        binary_mask = np.where(gray_image > 1, 255, 0).astype(np.uint8)

        rows_with_content = np.any(binary_mask, axis=1)
        cols_with_content = np.any(binary_mask, axis=0)

        if np.any(rows_with_content) and np.any(cols_with_content):
            y_min, y_max = np.where(rows_with_content)[0][[0, -1]]
            x_min, x_max = np.where(cols_with_content)[0][[0, -1]]
            
            return img[y_min:y_max + 1, x_min:x_max + 1]
        
        return img

    def warp_image(self, img, H, output_shape):
        output_img = np.zeros((output_shape[1], output_shape[0], img.shape[2]), dtype=img.dtype)
        H_inv = np.linalg.inv(H)

        for y in range(output_shape[1]):
            for x in range(output_shape[0]):
                p = np.array([x, y, 1])
                src_coords = H_inv @ p
                src_coords /= src_coords[2]  # Normalize
                
                src_x, src_y = src_coords[0], src_coords[1]
                
                if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                    x0, y0 = int(src_x), int(src_y)
                    x1, y1 = min(x0 + 1, img.shape[1] - 1), min(y0 + 1, img.shape[0] - 1)

                    a = src_x - x0
                    b = src_y - y0

                    output_img[y, x] = (
                        (1 - a) * (1 - b) * img[y0, x0] +
                        a * (1 - b) * img[y0, x1] +
                        (1 - a) * b * img[y1, x0] +
                        a * b * img[y1, x1]
                    )
        
        return output_img

    def determine_output_size(self, H_matrices, anchor_shape):
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for H in H_matrices:
            corners = np.array([[0, 0, 1], [anchor_shape[1], 0, 1],
                                [anchor_shape[1], anchor_shape[0], 1], [0, anchor_shape[0], 1]])
            transformed_corners = H @ corners.T
            transformed_corners /= transformed_corners[2, :]  # Normalize

            min_x = min(min_x, transformed_corners[0, :].min())
            max_x = max(max_x, transformed_corners[0, :].max())
            min_y = min(min_y, transformed_corners[1, :].min())
            max_y = max(max_y, transformed_corners[1, :].max())

        width = int(max_x - min_x)
        height = int(max_y - min_y)
        return (width, height)
    
    def make_panaroma_for_images_in(self, path):
        image_files = sorted(glob.glob(path + os.sep + '*'))
        images = [cv2.imread(img_file) for img_file in image_files]

        mid_index = len(images) // 2
        anchor_image = images[mid_index]
        diagonal_length = np.sqrt(anchor_image.shape[0] ** 2 + anchor_image.shape[1] ** 2)
        focal_length = diagonal_length / np.tan(np.pi / 5)

        cylindrical_images = [self.apply_cylindrical_projection(img, focal_length) for img in images]

        H_matrices = [None] * len(cylindrical_images)
        H_matrices[mid_index] = np.eye(3)

        for i in range(mid_index - 1, -1, -1):
            H_matrices[i] = self.find_homography(cylindrical_images[i], cylindrical_images[i + 1]) @ H_matrices[i + 1]
            print(f"Calculated left homography for image {i}.")

        for i in range(mid_index + 1, len(cylindrical_images)):
            H_matrices[i] = np.linalg.inv(self.find_homography(cylindrical_images[i - 1], cylindrical_images[i])) @ H_matrices[i - 1]
            print(f"Calculated right homography for image {i}.")

        output_size = self.determine_output_size(H_matrices, anchor_image.shape)
        stitched_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

        for i in range(len(cylindrical_images)):
            transformed_image = self.warp_image(cylindrical_images[i], H_matrices[i], output_size)
            stitched_image = self.merge_images(stitched_image, transformed_image)
            print(f"Blended image {i} into the panorama.")

        return self.crop_to_content(stitched_image), H_matrices
