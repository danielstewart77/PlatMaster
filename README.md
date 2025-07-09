# PlatMaster

A sophisticated OCR and AI-powered document processing system for extracting structured data from oil and gas well location plats. PlatMaster combines advanced computer vision (PaddleOCR) with large language models (Azure OpenAI) to automatically extract critical drilling location information from PDF plat documents.

## 🌟 Features

- **Advanced OCR Processing**: Uses PaddleOCR with GPU acceleration for high-accuracy text detection and recognition
- **AI-Powered Data Extraction**: Leverages Azure OpenAI GPT models to extract structured data from OCR results
- **Multi-Format Support**: Processes PDF documents and converts them to images for analysis
- **Visual Debugging**: Generates annotated images showing detected text regions with confidence scores
- **Structured Output**: Extracts key drilling location data including coordinates, elevations, and survey points
- **Batch Processing**: Processes multiple plat documents automatically

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (optional, for faster OCR processing)
- Azure OpenAI API access

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PlatMaster
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_API_KEY_EAST2=your_east2_api_key_here
```

### Usage

1. Place your PDF plat documents in the `plats/` directory
2. Run the processing pipeline:
```bash
python main.py
```

3. Check the `output/` directory for results:
   - `*_input.png`: Original PDF pages converted to images
   - `*_boxes.png`: Images with OCR bounding boxes overlaid
   - `*_ocr_merged.txt`: Raw OCR text output
   - `*.json`: Structured extracted data

## 📊 Data Extraction

PlatMaster extracts the following structured data from plat documents:

### Location Points
- **Surface Hole Location (SHL)**: Primary drilling location with elevation
- **Penetration Point (PP)**: Point where drilling enters the target formation
- **First Take Point (FTP)**: Beginning of horizontal drilling section
- **Last Take Point (LTP)**: End of horizontal drilling section
- **Bottom Hole Location (BHL)**: Final drilling destination

### Coordinate Systems
- Supports NAD 27 and NAD 83 coordinate systems
- Automatically converts degree format to decimal format
- Extracts Texas Central Zone coordinates
- Handles various coordinate notation formats

### Sample Output
```json
{
  "elevation": "3025.3",
  "surface_hole_location": {
    "elevation": "3025.3",
    "lat": "32.4208806",
    "lon": "-102.3465417"
  },
  "penetration_point": {
    "lat": "32.4221000",
    "lon": "-102.3425111"
  },
  "first_take_point": {
    "lat": "32.4232833",
    "lon": "-102.3430583"
  },
  "last_take_point": {
    "lat": "32.4506389",
    "lon": "-102.3518278"
  },
  "bottom_hole_location": {
    "lat": "32.4506389",
    "lon": "-102.3518278"
  }
}
```

## 🔧 Technical Architecture

### Components

1. **PDF Processing** (`pdf2image`): Converts PDF documents to high-resolution images
2. **OCR Engine** (`PaddleOCR`): Detects and recognizes text with confidence scoring
3. **AI Extraction** (`Azure OpenAI`): Uses structured prompts to extract meaningful data
4. **Data Validation** (`Pydantic`): Ensures extracted data conforms to expected schemas

### OCR Configuration
- Language: English
- Device: GPU (fallback to CPU)
- Text orientation: Disabled for better performance on structured documents
- Document unwrapping: Disabled for plat-specific optimization

### AI Model Integration
- Primary model: GPT-4.1 (fine-tuned deployment)
- Fallback model: Standard GPT-4
- Response format: Structured JSON schema
- Temperature: Deterministic (seed: 7779)

## 📁 Project Structure

```
PlatMaster/
├── main.py                 # Main processing pipeline
├── models/
│   └── plat.py            # Pydantic data models
├── services/
│   └── llm.py             # Azure OpenAI integration
├── plats/                 # Input PDF documents
├── output/                # Generated results
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
└── README.md             # This file
```

## 🔍 Debugging and Troubleshooting

### Visual Debugging
The system generates annotated images (`*_boxes.png`) that show:
- Green bounding boxes around detected text regions
- Red text labels with detected content and confidence scores
- Visual verification of OCR accuracy

### Common Issues

1. **Low OCR Confidence**: Check image quality and resolution
2. **Missing Coordinates**: Verify plat format matches expected patterns
3. **API Errors**: Confirm Azure OpenAI credentials and deployment names
4. **GPU Issues**: System automatically falls back to CPU processing

### Logging
The system provides detailed logging for:
- OCR processing progress
- Text detection statistics
- LLM API interactions
- Error diagnosis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the GNU General Public License (GPL) - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PaddleOCR**: High-performance OCR engine
- **Azure OpenAI**: Advanced language model capabilities
- **Pydantic**: Data validation and settings management
- **pdf2image**: PDF to image conversion utilities

## 📞 Support

For questions, issues, or contributions, please open an issue on the GitHub repository or contact the development team.

---

Built with ❤️ for the oil and gas industry
