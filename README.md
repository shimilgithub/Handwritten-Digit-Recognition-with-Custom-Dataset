# Handwriting Recognition with Custom Dataset

## Overview
This project focuses on implementing and testing a handwriting recognition model using the MNIST dataset format. The goal is to create a class for recognizing handwritten digits, generate custom handwritten images, and validate the model through Python scripts and a Jupyter Notebook. This project demonstrates an understanding of machine learning concepts and practical implementation for digit recognition.

## Technology Used
- **Programming Language:** Python
- **Libraries and Tools:**
  - NumPy
  - Matplotlib
  - argparse/sys (for command-line argument parsing)
  - Jupyter Notebook (for documentation and testing)

## Dataset Used
- **Custom MNIST Samples:**
  - 10 digits (0-9) with five handwritten images per digit.
  - Image specifications:
    - Dimensions: `28 x 28 x 1` (grayscale).
    - Naming convention: `n_m.png` where:
      - `n`: Digit (0-9)
      - `m`: Index (0-4)
  - Example: `0_0.png`, `1_3.png`.

## Repository Organization
```
project-directory/
├── mnist.py          # Implementation of the MNIST class
├── module5-3.py      # Script for testing the handwriting recognition
├── module5-3.ipynb   # Jupyter Notebook to showcase your work
├── Custom MNIST Samples/           # Directory containing your custom handwritten images for testing
├── datase/ #Directory containing MNIST images for training
└── README.md         # Project documentation
```

## How to Run

### Prerequisites
- Python 3.x installed.
- Required libraries installed (NumPy, Matplotlib).

### Steps
1. **Set Up Handwritten Images:**
   - Place your custom handwritten images (five per digit) in the `images/` directory.
   - Ensure they follow the naming convention (`n_m.png`).

2. **Complete the MNIST Class:**
   - Implement the functionality in `mnist.py`.

3. **Run the Testing Script:**
   - Execute the script `module5-3.py` with the appropriate arguments:
     ```bash
     $ python module5-3.py <image_filename> <expected_digit>
     ```
     Example:
     ```bash
     $ python module5-3.py 3_2.png 3
     ```

4. **Validate Output:**
   - Success example:
     ```
     Success: Image 3_2.png is for digit 3 and recognized as 3.
     ```
   - Failure example:
     ```
     Fail: Image 3_2.png is for digit 3 but the inference result is 5.
     ```

5. **Run Jupyter Notebook:**
   - Open and execute `module5-3.ipynb` to:
     - Showcase your development process.
     - Test the functionality interactively.
   - Use the following command to execute Python scripts inside the notebook:
     ```python
     !python module5-3.py <image_filename> <expected_digit>
     ```

## Submission Checklist
- `mnist.py`: Completed implementation of the MNIST class.
- `module5-3.py`: Script for testing handwritten digit recognition.
- `module5-3.ipynb`: Jupyter Notebook showcasing your work.
- Handwritten images: Five images per digit (0-9) in the `Custom MNIST Samples/` directory.


