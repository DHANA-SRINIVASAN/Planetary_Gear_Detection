# Planetary Gear Detection

A computer vision application for detecting and counting planetary gears in video footage using template matching and non-maximum suppression.

## Features

- **Interactive Template Selection**: Select multiple regions of interest (ROI) from the first frame to create templates
- **Real-time Detection**: Process video frames with adjustable detection threshold
- **Non-Maximum Suppression**: Eliminate duplicate detections using NMS algorithm
- **Interactive Controls**: 
  - Adjustable detection threshold via trackbar
  - Frame-by-frame navigation with position trackbar
  - Real-time gear counting display
- **Template Management**: Automatically saves selected templates with timestamps
- **Output Video Generation**: Save processed video with detection annotations (optional)

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Video file containing planetary gears

## Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install opencv-python numpy
```

## Usage

1. **Setup**: Update the video path in `main.py`:
```python
video_path = r"path/to/your/video.mp4"
```

2. **Run the application**:
```bash
python main.py
```

3. **Template Selection**:
   - The application will display the first frame
   - Select multiple regions containing planetary gears by drawing rectangles
   - Press SPACE or ENTER to confirm each selection
   - Press ESC when finished selecting templates

4. **Detection Phase**:
   - Use the "Threshold" trackbar to adjust detection sensitivity (0.0 - 1.0)
   - Use the "Position" trackbar to navigate through video frames
   - Press 'q' to quit the application

5. **Output Video** (Optional):
   - Enable `save_output_video = True` in the configuration section
   - The processed video with detection annotations will be saved to the `output/` directory
   - Output video includes all detection rectangles and gear counts

## How It Works

### Template Matching
The application uses OpenCV's `matchTemplate()` function with normalized cross-correlation (`TM_CCOEFF_NORMED`) to find instances of the selected templates in each frame.

### Non-Maximum Suppression
To eliminate duplicate detections, the application implements a custom NMS algorithm that:
- Calculates overlap between detected bounding boxes
- Removes boxes with overlap above the threshold (0.3)
- Keeps the detection with the highest confidence

### Key Components

- **Template Selection**: Interactive ROI selection using `cv2.selectROIs()`
- **Detection Pipeline**: Template matching → Thresholding → NMS → Visualization
- **User Interface**: Trackbars for threshold adjustment and frame navigation

## File Structure

```
Planetary_Gear_Detection/
├── main.py              # Main application script
├── templates/           # Directory for saved templates
├── output/              # Directory for output videos (created automatically)
├── README.md           # This file
└── Planetary_Gear_Backup.mp4  # Video file (update path as needed)
```

## Configuration

Key parameters that can be adjusted in `main.py`:

- `video_path`: Path to your input video file
- `template_folder`: Directory to save template images
- `scale_percent_for_display`: Display scaling percentage (default: 60%)
- `overlapThresh`: NMS overlap threshold (default: 0.3)
- `save_output_video`: Enable/disable output video saving (optional feature)
- `output_video_path`: Path for the output video file (if enabled)

## Controls

- **Threshold Trackbar**: Adjust detection sensitivity (0-20, mapped to 0.0-1.0)
- **Position Trackbar**: Navigate through video frames
- **'q' Key**: Quit the application

## Output

- **Visual**: Green rectangles around detected gears
- **Count Display**: Real-time count of detected gears in the top-left corner
- **Templates**: Saved template images in the `templates/` directory with timestamps
- **Output Video**: Processed video with detection annotations (optional feature)

## Output Video Setup

To enable output video generation, modify the configuration section in `main.py`:

```python
# === CONFIG ===
video_path = r"path/to/your/video.mp4"
template_folder = r"templates"
output_folder = r"output"  # Add this line
save_output_video = True   # Add this line
scale_percent_for_display = 60
```

The output video will be saved with the same codec and frame rate as the input video, with detection annotations overlaid.

## Troubleshooting

1. **Video not loading**: Check the video path and ensure the file exists
2. **No detections**: Try lowering the threshold or selecting better templates
3. **Too many false positives**: Increase the threshold or improve template selection
4. **Performance issues**: Reduce the display scale percentage
5. **Output video not saving**: Ensure the output directory exists and you have write permissions

## Technical Details

- **Detection Method**: Normalized Cross-Correlation Template Matching
- **NMS Algorithm**: Custom implementation with area-based overlap calculation
- **Image Processing**: Grayscale conversion for template matching
- **UI Framework**: OpenCV's built-in GUI components

## License

This project is open source and available under the MIT License.