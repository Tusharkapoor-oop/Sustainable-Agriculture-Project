# 🌾 AgriSenseAI - Smart Sustainable Agriculture Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.33+-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

**AgriSenseAI** is an AI-powered sustainable agriculture assistant designed to help farmers make data-driven decisions. The platform combines computer vision, machine learning, and NLP to provide:

- **🔍 Crop Disease Detection** - Upload leaf images for instant AI diagnosis
- **🌱 Smart Crop Recommendations** - Get optimal crop suggestions based on soil and climate
- **💧 Intelligent Irrigation Advisory** - Optimize water usage with ML-powered recommendations  
- **📊 Sustainability Scoring** - Track and improve your eco-friendly farming practices
- **🤖 NLP Farm Assistant** - Ask questions and get expert agricultural advice

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   FastAPI       │────│   ML Models     │
│   (Port 8501)   │    │   (Port 8000)   │    │   (Local/Cloud) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
    ┌────▼────┐             ┌────▼────┐             ┌────▼────┐
    │ Pages & │             │ REST    │             │ Disease │
    │ Components│           │ Endpoints│            │ Crop    │
    └─────────┘             └─────────┘             │ Models  │
                                                    └─────────┘
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd AgriSenseAI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Start the application (both UI and API)
streamlit run app/main.py
```

The application will start with:
- **Streamlit UI**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 3. Test the Installation

```bash
# Run unit tests
pytest -q

# Check API health
curl http://localhost:8000/api/health
```

## 📁 Project Structure

```
AgriSenseAI/
├── app/                     # Frontend application
│   ├── main.py             # Main Streamlit app with FastAPI integration
│   ├── components/         # Reusable UI components
│   ├── pages/             # Individual app pages
│   ├── style.css          # Custom styling and animations
│   └── animations/        # Lottie animation files
├── core/                   # ML and business logic
│   ├── disease_model.py   # CNN disease classification
│   ├── crop_model.py      # Crop recommendation engine
│   ├── irrigation_model.py # Smart irrigation advisor
│   ├── nlp_advisor.py     # NLP-powered farm assistant
│   └── sustainability.py  # Eco-score calculator
├── models/                 # Trained ML models (optional)
├── data/                   # Datasets
├── tests/                  # Unit tests
├── docs/                   # Documentation and presentations
└── assets/                 # Images and icons
```

## 🔧 API Endpoints

### Disease Detection
```http
POST /api/disease
Content-Type: multipart/form-data

Response: {
  "label": "Early Blight",
  "confidence": 0.87,
  "remedy": "Apply copper-based fungicide...",
  "notes": "Monitor weekly for spread"
}
```

### Crop Recommendation
```http
POST /api/recommend_crop
Content-Type: application/json

{
  "nitrogen": 40,
  "phosphorus": 67,
  "potassium": 60,
  "ph": 6.2,
  "rainfall": 180,
  "region": "Karnataka",
  "season": "Kharif"
}

Response: {
  "crop": "Rice",
  "score": 0.92,
  "reason": "Optimal soil nutrients and monsoon rainfall"
}
```

### Irrigation Advisory
```http
POST /api/irrigation
Content-Type: application/json

{
  "crop": "Tomato",
  "stage": "Flowering",
  "soil_moisture": 0.3,
  "temperature": 28.5,
  "humidity": 65,
  "wind": 12
}

Response: {
  "water_l_per_hectare": 450.0,
  "schedule_hours": 6.0,
  "note": "Water early morning to reduce evaporation"
}
```

## 🎨 Features

### Smart Stubs System
- **Graceful Fallbacks**: App works perfectly without trained models
- **Auto-Detection**: Automatically loads real models if present in `/models`
- **Deterministic Output**: Consistent results for demos and testing

### User Experience
- **One-Command Start**: Single command starts both UI and API
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Feedback**: Loading states, progress bars, and success messages
- **Error Handling**: Friendly error messages with troubleshooting steps

### Sustainability Focus
- **Eco-Score Tracking**: Monitor water and chemical usage
- **Regional Adaptation**: Location-aware recommendations
- **Best Practices**: Actionable tips for sustainable farming

## 📊 Model Integration

### Optional ML Models
Place trained models in the `/models` directory:

```
models/
├── cnn_disease.h5        # TensorFlow/Keras disease classifier
├── crop_recommender.pkl  # Scikit-learn crop recommendation
└── irrigation.pkl        # Regression model for irrigation
```

### Model Requirements
- **Disease Model**: Keras .h5 file with input shape (224, 224, 3)
- **Crop Model**: Scikit-learn pipeline with joblib serialization
- **Irrigation Model**: Regression model accepting 6 features

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_crop.py -v

# Run with coverage
pytest --cov=core --cov=app

# Test API endpoints
pytest tests/test_disease.py::test_disease_endpoint
```

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Kill processes on ports 8000/8501
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9
```

**Model Loading Errors**
- Check model file format and location in `/models`
- Verify model input/output shapes match expected format
- Review logs for detailed error messages

**Dependency Conflicts**
```bash
# Clean install
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
```

## 📈 Performance

- **API Response Time**: < 200ms for predictions
- **Image Processing**: < 2s for disease detection
- **Memory Usage**: < 500MB with all models loaded
- **Concurrent Users**: Supports 10+ simultaneous requests

## 🌱 Environmental Variables

```bash
# Optional configuration
export HF_HOME=".cache/huggingface"
export TRANSFORMERS_OFFLINE="0"  
export AGRI_LOCALE_DEFAULT="en"
```

## 📝 Development

### Adding New Models
1. Place model file in `/models` directory
2. Update corresponding core module (e.g., `core/disease_model.py`)
3. Add tests in `tests/` directory
4. Update API documentation

### Custom Styling
- Edit `app/style.css` for theme customization
- Add Lottie animations in `app/animations/`
- Update component styling in individual files

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-model`)
3. Commit changes (`git commit -am 'Add new crop model'`)
4. Push to branch (`git push origin feature/new-model`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PlantVillage Dataset for disease classification
- Indian Agricultural Research Institute for crop data
- OpenWeatherMap API for weather integration
- Streamlit and FastAPI communities for excellent documentation

## 📞 Support

For issues and questions:
- Create GitHub Issue
- Email:tusharkapoor052@gmail.com
- Documentation: [docs/project_report.md](docs/project_report.md)

---

**Built  for sustainable agriculture**
