# Low-Light Image Enhancer

This project is a **deep learning-based image enhancement tool** designed to improve visibility in low-light images.  
It fuses the outputs of multiple specialized models to brighten dark regions, preserve natural mid-tones, and prevent overexposure in bright areas.

## Features
- Upload and enhance low-light images directly from the web interface  
- Fusion of three models (`real_mass.h5`, `real_dream.h5`, `finalmass.h5`) for adaptive enhancement  
- Real-time preview of original vs. enhanced images  
- Side-by-side comparison mode  
- Download enhanced images instantly  

## Tech Stack
- **Frontend/UI**: [Streamlit](https://streamlit.io/)  
- **Deep Learning**: TensorFlow / Keras  
- **Image Processing**: OpenCV, NumPy, scikit-image  
- **Other Utilities**: PIL, time  

## How It Works
1. **Upload an image** (JPG/PNG) through the Streamlit app.  
2. The system resizes the image to `256x256` and processes it through three pre-trained models:  
   - **Low-light model** – Enhances very dark areas.  
   - **Mid-light model** – Handles balanced lighting.  
   - **Bright model** – Preserves highlights.  
3. A **fusion mechanism** computes masks based on brightness levels and blends model outputs adaptively.  
4. The final enhanced image is displayed and can be downloaded.  

## Project Structure

├── final_g.ipynb # Jupyter notebook (experiments & model analysis)

├── mass.py # Streamlit app for enhancement

├── real_mass.h5 # Trained model (low-light)

├── real_dream.h5 # Trained model (mid-light)

├── finalmass.h5 # Trained model (bright regions)

└── README.md # Project documentation


## Usage
1. Clone the repository
```bash
git clone https://github.com/your-username/low-light-enhancer.git
cd low-light-enhancer
```

2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the app
```bash
streamlit run mass.py
```

4. Upload & Enhance

    Upload your low-light image.

    Click Enhance Image to generate the improved version.

    Download or compare with the original.

📊 Metrics (Optional)

The code also supports image quality metrics such as:

    PSNR (Peak Signal-to-Noise Ratio)

    MSE (Mean Squared Error)

    SSIM (Structural Similarity Index)
