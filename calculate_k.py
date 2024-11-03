import cv2
import numpy as np
from pathlib import Path
from PIL import Image

class ChessboardCalibration:
    def __init__(self, chessboard_size, pattern_dir, output_dir, output_file):
        self.chessboard_size = chessboard_size
        self.pattern_images = list(Path(pattern_dir).glob('*.png'))
        self.output_dir = output_dir
        self.output_file = output_file
        self.world_points = self._prepare_world_points()
        self.all_world_points = []
        self.all_image_points = []

    def _prepare_world_points(self):
        points = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), dtype=np.float32)
        points[:, :2] = np.indices((self.chessboard_size[1], self.chessboard_size[0])).T.reshape(-1, 2)
        return points

    def process_images(self):
        for image_path in self.pattern_images:
            image = self._load_image(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found, corners = self._find_corners(gray_image)

            if found:
                self.all_world_points.append(self.world_points)
                self.all_image_points.append(corners)
                self._visualize_and_save_corners(image, corners, image_path.stem)

    def _load_image(self, image_path):
        pil_image = Image.open(image_path)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _find_corners(self, gray_image):
        return cv2.findChessboardCorners(gray_image, self.chessboard_size, None)

    def _visualize_and_save_corners(self, image, corners, image_name):
        image_with_corners = cv2.drawChessboardCorners(image, self.chessboard_size, corners, True)
        output_path = Path(self.output_dir, f'{image_name}_corners.png')
        cv2.imwrite(str(output_path), image_with_corners)

    def calibrate_camera(self):
        calibration_success, camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors = cv2.calibrateCamera(
            self.all_world_points, self.all_image_points, (self.chessboard_size[1], self.chessboard_size[0]), None, None)
        if calibration_success:
            np.savez(self.output_file, K=camera_matrix, dist=distortion_coeffs, rvecs=rotation_vectors, tvecs=translation_vectors)

def main():
    chessboard_size = (6, 9)
    pattern_dir = 'patterns'
    output_dir = './patterns'
    output_file = 'K.npz'

    calibration = ChessboardCalibration(chessboard_size, pattern_dir, output_dir, output_file)
    calibration.process_images()
    calibration.calibrate_camera()

if __name__ == "__main__":
    main()
