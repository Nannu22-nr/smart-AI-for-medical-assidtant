import os
import logging
import json
import re
import io
import base64
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Optional, Dict, Any, List

from google import genai
from google.genai import types
from PIL import Image
import pandas as pd
from pypdf import PdfReader
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("❌ Missing GEMINI_API_KEY. Please set it in environment variables.")
else:
    logger.info("✅ GEMINI_API_KEY detected")
    # Initialize the new GenAI client
    client = genai.Client(api_key=GEMINI_API_KEY)

# Flask App Configuration
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
CORS(app, resources={
    r"/*": {"origins": "*"}
})

# Generate a random secret key if not provided
app.secret_key = os.getenv("FLASK_SECRET_KEY", str(uuid4()))
app.permanent_session_lifetime = timedelta(hours=4)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf', 'csv'}
MAX_FILE_SIZE = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Gemini Model Configuration
MODEL_NAME = "gemini-2.5-flash"
VISION_MODEL_NAME = "gemini-2.5-flash"
MAX_CONVERSATION_HISTORY = 100

# System Prompts
SYSTEM_PROMPT_CHAT = (
    "You are Nannu, an advanced AI medical assistant with expertise in general medicine, "
    "diagnostics, and patient care. When responding:\n"
    "• Always use clear bullet points (*) and numbered lists\n"
    "• Separate main ideas with paragraph breaks\n"
    "• For urgent symptoms, clearly state 'URGENT: Seek immediate medical attention'\n"
    "• Provide severity assessment (Low/Medium/High concern)\n"
    "• Ask relevant follow-up questions\n"
    "• Include helpful health tips when appropriate\n"
    "• Always end with: 'This is AI guidance, not a professional diagnosis. Consult a healthcare provider.'\n"
    "Be empathetic, thorough, and prioritize patient safety."
)

SYSTEM_PROMPT_SYMPTOMS = (
    "You are a medical symptom analyzer. Analyze the provided symptoms and provide:\n"
    "1. **Possible Conditions** (most likely to least likely)\n"
    "2. **Severity Assessment** (Low/Medium/High/Emergency)\n"
    "3. **Recommended Actions**\n"
    "4. **When to Seek Care** (timeline)\n"
    "5. **Warning Signs** to watch for\n"
    "Format with clear headers and bullet points. Always prioritize safety."
)

SYSTEM_PROMPT_NUTRITION = (
    "You are NutriBot, an advanced nutrition AI. Create detailed, personalized nutrition plans with:\n"
    "• **Daily Meal Plans** (breakfast, lunch, dinner, snacks)\n"
    "• **Macronutrient Breakdown** (carbs, protein, fats)\n"
    "• **Calorie Targets** based on goals\n"
    "• **Supplement Recommendations**\n"
    "• **Hydration Guidelines**\n"
    "• **Shopping List**\n"
    "Always include: 'Consult a registered dietitian for personalized advice.'"
)

SYSTEM_PROMPT_DRUG = (
    "You are a pharmaceutical interaction analyzer. For the provided medications:\n"
    "1. **Interaction Risk Level** (None/Low/Medium/High/Dangerous)\n"
    "2. **Specific Interactions** (detailed explanations)\n"
    "3. **Timing Recommendations**\n"
    "4. **Food Interactions**\n"
    "5. **Monitoring Advice**\n"
    "Always emphasize consulting a pharmacist or doctor."
)

SYSTEM_PROMPT_MEDICAL_IMAGE = (
    "You are a medical imaging AI specialist. Analyze the provided medical image (X-ray, MRI, or CT scan) and provide:\n"
    "1. **Image Type Assessment** (what type of scan this appears to be)\n"
    "2. **Image Quality** (diagnostic quality assessment)\n"
    "3. **Primary Findings** (what you observe in the image)\n"
    "4. **Abnormalities Detected** (if any)\n"
    "5. **Possible Conditions** (differential diagnosis)\n"
    "6. **Severity Assessment** (Low/Medium/High/Emergency)\n"
    "7. **Recommended Next Steps**\n"
    "8. **Urgency Level**\n"
    "Format with clear headers and bullet points. Always include: 'This AI analysis is for informational purposes only. "
    "All medical images must be reviewed by a qualified radiologist or healthcare provider for definitive diagnosis.'"
)

SYSTEM_PROMPT_HEALTH_RECORD = (
    "You are a health record analyzer. Analyze the provided health report data and provide:\n"
    "1. **Summary of Key Findings**\n"
    "2. **Identified Abnormalities** (with explanations)\n"
    "3. **Potential Health Concerns**\n"
    "4. **Suggested Follow-up Questions** for the patient or doctor\n"
    "5. **Recommendations**\n"
    "Format with clear headers and bullet points. Always prioritize safety and emphasize consulting a healthcare professional."
)

SYSTEM_PROMPT_RISK_TIPS = (
    "You are a health risk advisor. Based on the provided risk type, score, and patient details, provide:\n"
    "1. **Risk Interpretation**\n"
    "2. **Personalized Lifestyle Tips**\n"
    "3. **Preventive Measures**\n"
    "4. **When to Consult a Doctor**\n"
    "Format with clear headers and bullet points. Be encouraging and focus on actionable advice."
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_input(text):
    if not text:
        return ""
    text = re.sub(r'[<>"\']', '', text)
    return text[:2000]

def format_ai_response(text):
    if not text:
        return text
    
    if not any(marker in text for marker in ['*', '-', '•', '\n1.', '\n2.']):
        sentences = text.split('. ')
        if len(sentences) > 1:
            text = '\n• ' + '\n• '.join(sentence.strip() + '.' for sentence in sentences if sentence.strip())
    
    return text

def get_conversation_summary():
    if "conversation_history" not in session:
        return "No conversation yet."
    
    messages = session["conversation_history"][1:]
    if not messages:
        return "No conversation yet."
    
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    return f"Topics discussed: {', '.join(user_messages[-3:])}"

def analyze_medical_image(image_data, image_type="medical_scan"):
    """Analyze medical image using Gemini Vision API"""
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    
    try:
        # Open the image
        image = Image.open(io.BytesIO(image_data))
        
        # Prepare the prompt
        prompt = f"""
        You are a medical imaging AI specialist. Analyze this {image_type} image carefully.
        
        Please provide:
        1. **Image Type Assessment**: What type of scan is this? (X-ray, MRI, CT, Ultrasound, etc.)
        2. **Image Quality**: Is the image diagnostic quality?
        3. **Primary Findings**: Describe what you observe in the image
        4. **Abnormalities Detected**: List any abnormal findings
        5. **Possible Conditions**: Differential diagnosis based on imaging findings
        6. **Severity Assessment**: Low/Medium/High/Emergency
        7. **Urgency Level**: When should the patient seek care?
        8. **Recommended Next Steps**: What should the patient do?
        
        Important disclaimers:
        - This is an AI-assisted analysis and not a definitive diagnosis
        - All medical images must be reviewed by a qualified radiologist
        - Some findings may be subtle or missed by AI
        - Clinical correlation is essential
        
        Be thorough, professional, and prioritize patient safety.
        """
        
        # Generate response using the new GenAI client
        response = client.models.generate_content(
            model=VISION_MODEL_NAME,
            contents=[prompt, image]
        )
        
        return {
            "success": True,
            "analysis": response.text,
            "image_type": image_type,
            "model_used": VISION_MODEL_NAME
        }
        
    except Exception as e:
        logger.error(f"Error in medical image analysis: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}

def calculate_diabetes_risk(data):
    """Calculate diabetes risk score"""
    score = 0
    age = data.get('age', 0)
    bmi = data.get('bmi', 0)
    waist = data.get('waist_circumference', 0)
    physical_activity = data.get('physical_activity', False)
    veggies = data.get('daily_veggies_fruit', False)
    hypertension_med = data.get('hypertension_med', False)
    high_blood_sugar = data.get('high_blood_sugar_history', False)
    family_diabetes = data.get('family_diabetes', False)

    if age >= 45 and age < 55: score += 2
    elif age >= 55 and age < 65: score += 3
    elif age >= 65: score += 4

    if bmi >= 25 and bmi < 30: score += 1
    elif bmi >= 30: score += 3

    if (data.get('gender', 'male') == 'male' and waist >= 94) or (data.get('gender', 'female') == 'female' and waist >= 80):
        score += 3
    if (data.get('gender', 'male') == 'male' and waist >= 102) or (data.get('gender', 'female') == 'female' and waist >= 88):
        score += 1

    if not physical_activity: score += 2
    if not veggies: score += 1
    if hypertension_med: score += 2
    if high_blood_sugar: score += 5
    if family_diabetes: score += 5
    elif family_diabetes == 'distant': score += 3

    if score < 7: risk_level = 'Low'
    elif score < 12: risk_level = 'Slightly Elevated'
    elif score < 15: risk_level = 'Moderate'
    elif score < 20: risk_level = 'High'
    else: risk_level = 'Very High'

    return {'score': score, 'risk_level': risk_level}

def calculate_heart_disease_risk(data):
    """Calculate heart disease risk score"""
    score = 0
    age = data.get('age', 0)
    gender = data.get('gender', 'male')
    cholesterol = data.get('total_cholesterol', 0)
    hdl = data.get('hdl_cholesterol', 0)
    systolic_bp = data.get('systolic_bp', 0)
    smoker = data.get('smoker', False)
    diabetes = data.get('diabetes', False)
    treated_bp = data.get('treated_hypertension', False)

    if gender == 'male':
        if age >= 45 and age < 55: score += 3
        elif age >= 55: score += 6
    else:
        if age >= 50 and age < 60: score += 4
        elif age >= 60: score += 7

    if cholesterol >= 200 and cholesterol < 240: score += 1
    elif cholesterol >= 240: score += 2

    if hdl < 40: score += 2
    elif hdl >= 60: score -= 1

    if systolic_bp >= 130 and systolic_bp < 140: score += 1
    elif systolic_bp >= 140: score += 2
    if treated_bp: score += 1

    if smoker: score += 2
    if diabetes: score += 2

    if score < 3: risk_level = 'Low'
    elif score < 6: risk_level = 'Moderate'
    else: risk_level = 'High'

    return {'score': score, 'risk_level': risk_level}

def calculate_kidney_disease_risk(data):
    """Calculate kidney disease risk score"""
    score = 0
    age = data.get('age', 0)
    gender = data.get('gender', 'male')
    hypertension = data.get('hypertension', False)
    diabetes = data.get('diabetes', False)
    bmi = data.get('bmi', 0)
    smoker = data.get('smoker', False)
    anemia = data.get('anemia', False)
    proteinuria = data.get('proteinuria', False)

    if age >= 50: score += 2
    if age >= 70: score += 2

    if gender == 'female': score += 1

    if hypertension: score += 3
    if diabetes: score += 3
    if bmi >= 30: score += 2
    if smoker: score += 1
    if anemia: score += 2
    if proteinuria: score += 4

    if score < 3: risk_level = 'Low'
    elif score < 7: risk_level = 'Moderate'
    else: risk_level = 'High'

    return {'score': score, 'risk_level': risk_level}

# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_session", methods=["POST"])
def start_session():
    session.permanent = True
    session["conversation_history"] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    session["chat_metadata"] = {
        "start_time": datetime.now().isoformat(),
        "message_count": 0,
        "topics": []
    }
    logger.info("🆕 New session started")
    return jsonify({"ok": True, "message": "New session started"})

@app.route("/chat", methods=["POST"])
def chat():
    if not GEMINI_API_KEY:
        return jsonify({"ok": False, "reply": "Server not configured. Please set GEMINI_API_KEY."}), 500
    
    if "conversation_history" not in session:
        session["conversation_history"] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    
    user_input = sanitize_input((request.json or {}).get("message", "").strip())
    chat_mode = (request.json or {}).get("mode", "normal")
    
    if not user_input:
        return jsonify({"ok": False, "reply": "Please type a message."}), 400
    
    try:
        if "chat_metadata" not in session:
            session["chat_metadata"] = {"start_time": datetime.now().isoformat(), "message_count": 0, "topics": []}
        
        session["chat_metadata"]["message_count"] += 1
        
        system_prompt = SYSTEM_PROMPT_CHAT
        if chat_mode == "detailed":
            system_prompt += "\n\nProvide detailed, comprehensive responses with additional medical context."
        elif chat_mode == "quick":
            system_prompt += "\n\nProvide concise, direct responses focusing on key points."
        
        session["conversation_history"].append({"role": "user", "content": user_input})
        
        if len(session["conversation_history"]) > MAX_CONVERSATION_HISTORY + 1:
            session["conversation_history"] = [session["conversation_history"][0]] + session["conversation_history"][-MAX_CONVERSATION_HISTORY:]
        
        # Build conversation for Gemini
        conversation_content = system_prompt + "\n\n"
        for msg in session["conversation_history"][1:]:  # Skip system message
            if msg["role"] == "user":
                conversation_content += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation_content += f"Assistant: {msg['content']}\n"
        
        # Set temperature based on mode
        temperature = 0.3 if chat_mode == "detailed" else 0.5 if chat_mode == "quick" else 0.4
        
        # Generate response using the new GenAI client
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=conversation_content,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=800 if chat_mode == "detailed" else 400 if chat_mode == "quick" else 600,
            )
        )
        
        reply = response.text.strip()
        reply = format_ai_response(reply)
        
        session["conversation_history"].append({"role": "assistant", "content": reply})
        session.modified = True
        
        response_metadata = {
            "mode": chat_mode,
            "timestamp": datetime.now().isoformat(),
            "message_id": str(uuid4())[:8]
        }
        
        logger.info(f"✅ Chat response sent (mode: {chat_mode})")
        return jsonify({
            "ok": True, 
            "reply": reply, 
            "metadata": response_metadata,
            "conversation_summary": get_conversation_summary()
        })
        
    except Exception as e:
        logger.error(f"🔥 Gemini error in /chat: {str(e)}")
        return jsonify({"ok": False, "reply": f"Sorry, server error: {str(e)}"}), 500

@app.route("/analyze_medical_image", methods=["POST"])
def analyze_medical_image_endpoint():
    """Analyze medical images (X-ray, MRI, CT scan) using Gemini Vision API"""
    if not GEMINI_API_KEY:
        return jsonify({
            "ok": False, 
            "reply": "Gemini API key is not configured. Please contact administrator."
        }), 503
    
    try:
        image_data = None
        image_type = request.form.get('image_type', 'medical_scan')
        
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '':
                if allowed_file(file.filename):
                    image_data = file.read()
                else:
                    return jsonify({
                        "ok": False, 
                        "reply": "Invalid file format. Please upload PNG, JPG, JPEG, GIF, BMP, or TIFF files."
                    }), 400
        
        elif request.json and 'image_data' in request.json:
            try:
                image_data = base64.b64decode(request.json['image_data'])
            except Exception as e:
                return jsonify({
                    "ok": False, 
                    "reply": "Invalid base64 image data."
                }), 400
        
        if not image_data:
            return jsonify({
                "ok": False, 
                "reply": "No image provided. Please upload a medical image (X-ray, MRI, CT scan)."
            }), 400
        
        # Analyze the medical image
        analysis_result = analyze_medical_image(image_data, image_type)
        
        if "error" in analysis_result:
            return jsonify({
                "ok": False, 
                "reply": f"Analysis error: {analysis_result['error']}"
            }), 500
        
        response_data = {
            "ok": True,
            "analysis": analysis_result["analysis"],
            "image_type": image_type,
            "analysis_type": "medical_image_analysis",
            "timestamp": datetime.now().isoformat(),
            "model_used": analysis_result.get("model_used", VISION_MODEL_NAME),
            "disclaimer": "This AI analysis is for informational purposes only and should not replace professional medical diagnosis. All medical images must be reviewed by a qualified radiologist or healthcare provider for definitive diagnosis and treatment planning."
        }
        
        logger.info(f"✅ Medical image analysis completed for {image_type}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"🔥 Error in medical image analysis endpoint: {str(e)}")
        return jsonify({
            "ok": False, 
            "reply": f"An error occurred during image analysis: {str(e)}"
        }), 500

@app.route("/analyze_symptoms", methods=["POST"])
def analyze_symptoms():
    if not GEMINI_API_KEY:
        return jsonify({"ok": False, "reply": "Server not configured."}), 500
    
    data = request.json or {}
    symptoms = sanitize_input(data.get("symptoms", ""))
    duration = sanitize_input(data.get("duration", ""))
    severity = data.get("severity", "medium")
    age = data.get("age", "")
    gender = data.get("gender", "")
    
    if not symptoms:
        return jsonify({"ok": False, "reply": "Please describe your symptoms."}), 400
    
    try:
        prompt = f"""
        {SYSTEM_PROMPT_SYMPTOMS}

        Analyze these symptoms for a {age}-year-old {gender}:
        
        Symptoms: {symptoms}
        Duration: {duration}
        Patient-reported severity: {severity}
        
        Provide a comprehensive analysis following the guidelines.
        """
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=800,
            )
        )
        
        reply = response.text.strip()
        reply = format_ai_response(reply)
        
        logger.info("✅ Symptom analysis completed")
        return jsonify({
            "ok": True, 
            "reply": reply,
            "analysis_type": "symptoms",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"🔥 Error in symptom analysis: {str(e)}")
        return jsonify({"ok": False, "reply": f"Error analyzing symptoms: {str(e)}"}), 500

@app.route("/drug_interaction", methods=["POST"])
def drug_interaction():
    if not GEMINI_API_KEY:
        return jsonify({"ok": False, "reply": "Server not configured."}), 500
    
    data = request.json or {}
    medications = sanitize_input(data.get("medications", ""))
    allergies = sanitize_input(data.get("allergies", ""))
    
    if not medications:
        return jsonify({"ok": False, "reply": "Please list your medications."}), 400
    
    try:
        prompt = f"""
        {SYSTEM_PROMPT_DRUG}

        Check for drug interactions between these medications:
        {medications}
        
        Known allergies: {allergies if allergies else "None reported"}
        
        Provide detailed interaction analysis and safety recommendations.
        """
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=700,
            )
        )
        
        reply = response.text.strip()
        reply = format_ai_response(reply)
        
        logger.info("✅ Drug interaction analysis completed")
        return jsonify({
            "ok": True, 
            "reply": reply,
            "analysis_type": "drug_interaction",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"🔥 Error in drug interaction analysis: {str(e)}")
        return jsonify({"ok": False, "reply": f"Error analyzing drug interactions: {str(e)}"}), 500

@app.route("/nutrition", methods=["POST"])
def nutrition():
    if not GEMINI_API_KEY:
        return jsonify({"ok": False, "reply": "Server not configured."}), 500
    
    data = request.json or {}
    required = ["age", "weight", "height", "goal", "duration"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"ok": False, "reply": f"Missing: {', '.join(missing)}"}), 400
    
    try:
        age, weight, height = float(data["age"]), float(data["weight"]), float(data["height"])
        activity_level = data.get("activity_level", "moderate")
        dietary_restrictions = sanitize_input(data.get("dietary_restrictions", "none"))
        medical_conditions = sanitize_input(data.get("medical_conditions", "none"))
        
        prompt = f"""
        {SYSTEM_PROMPT_NUTRITION}

        Create a comprehensive nutrition plan for:
        - Age: {age} years
        - Weight: {weight}kg, Height: {height}cm
        - Goal: {data['goal']} over {data['duration']}
        - Activity Level: {activity_level}
        - Dietary Restrictions: {dietary_restrictions}
        - Medical Conditions: {medical_conditions}
        
        Include detailed meal plans, recipes, and shopping lists.
        """
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=1000,
            )
        )
        
        reply = response.text.strip()
        reply = format_ai_response(reply)
        
        logger.info("✅ Enhanced nutrition plan generated")
        return jsonify({
            "ok": True, 
            "reply": reply,
            "plan_type": "nutrition",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"🔥 Error in nutrition planning: {str(e)}")
        return jsonify({"ok": False, "reply": f"Error generating nutrition plan: {str(e)}"}), 500

@app.route("/upload_health_record", methods=["POST"])
def upload_health_record():
    if not GEMINI_API_KEY:
        return jsonify({"ok": False, "reply": "Server not configured."}), 500

    if 'file' not in request.files:
        return jsonify({"ok": False, "reply": "No file provided."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"ok": False, "reply": "No selected file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "reply": "Invalid file format. Please upload PDF or CSV."}), 400

    file_path = None
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        report_data = ""
        file_ext = filename.rsplit('.', 1)[1].lower()

        if file_ext == 'pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                report_data += page.extract_text() + "\n"
        elif file_ext == 'csv':
            df = pd.read_csv(file_path)
            report_data = df.to_string(index=False)

        if not report_data.strip():
            return jsonify({"ok": False, "reply": "Empty or unreadable file."}), 400

        prompt = f"""
        {SYSTEM_PROMPT_HEALTH_RECORD}

        Analyze this health report (blood test, prescription, etc.):
        
        Report Content:
        {report_data[:5000]}
        
        Summarize abnormalities and suggest follow-up questions.
        """

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=800,
            )
        )

        analysis = response.text.strip()
        analysis = format_ai_response(analysis)

        logger.info("✅ Health record analysis completed")
        return jsonify({
            "ok": True,
            "analysis": analysis,
            "analysis_type": "health_record",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"🔥 Error in health record upload/analysis: {str(e)}")
        return jsonify({"ok": False, "reply": f"Error processing health record: {str(e)}"}), 500
    
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    if not GEMINI_API_KEY:
        return jsonify({"ok": False, "reply": "Server not configured."}), 500

    data = request.json or {}
    risk_type = data.get("type")
    if not risk_type:
        return jsonify({"ok": False, "reply": "Missing risk type (diabetes, heart, kidney)."}), 400

    try:
        if risk_type == "diabetes":
            risk_result = calculate_diabetes_risk(data)
        elif risk_type == "heart":
            risk_result = calculate_heart_disease_risk(data)
        elif risk_type == "kidney":
            risk_result = calculate_kidney_disease_risk(data)
        else:
            return jsonify({"ok": False, "reply": "Invalid risk type. Choose diabetes, heart, or kidney."}), 400

        prompt = f"""
        {SYSTEM_PROMPT_RISK_TIPS}

        Provide lifestyle tips for {risk_type} risk:
        
        Risk Level: {risk_result['risk_level']}
        Score: {risk_result['score']}
        Patient Details: {json.dumps({k: v for k, v in data.items() if k != 'type'}, indent=2)}
        
        Focus on preventive measures and tips.
        """

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=600,
            )
        )

        tips = response.text.strip()
        tips = format_ai_response(tips)

        logger.info(f"✅ {risk_type.capitalize()} risk prediction completed: {risk_result['risk_level']}")
        return jsonify({
            "ok": True,
            "risk_result": risk_result,
            "lifestyle_tips": tips,
            "risk_type": risk_type,
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "This is a simplified risk assessment. Consult a healthcare professional for accurate evaluation."
        })

    except Exception as e:
        logger.error(f"🔥 Error in risk prediction: {str(e)}")
        return jsonify({"ok": False, "reply": f"Error predicting risk: {str(e)}"}), 500

@app.route("/export_chat", methods=["POST"])
def export_chat():
    if "conversation_history" not in session:
        return jsonify({"ok": False, "message": "No conversation to export."}), 400
    
    try:
        chat_data = {
            "export_date": datetime.now().isoformat(),
            "session_metadata": session.get("chat_metadata", {}),
            "conversation": []
        }
        
        for msg in session["conversation_history"][1:]:
            if msg["role"] in ["user", "assistant"]:
                chat_data["conversation"].append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": datetime.now().isoformat()
                })
        
        return jsonify({
            "ok": True, 
            "chat_data": chat_data,
            "total_messages": len(chat_data["conversation"])
        })
        
    except Exception as e:
        logger.error(f"🔥 Error exporting chat: {str(e)}")
        return jsonify({"ok": False, "message": f"Error exporting chat: {str(e)}"}), 500

@app.route("/chat_statistics", methods=["GET"])
def chat_statistics():
    if "conversation_history" not in session:
        return jsonify({"ok": False, "stats": None})
    
    try:
        history = session["conversation_history"][1:]
        user_messages = [msg for msg in history if msg["role"] == "user"]
        bot_messages = [msg for msg in history if msg["role"] == "assistant"]
        
        stats = {
            "total_messages": len(history),
            "user_messages": len(user_messages),
            "bot_messages": len(bot_messages),
            "session_duration": "Active",
            "topics_discussed": len(set(msg["content"][:50] for msg in user_messages[-10:]))
        }
        
        return jsonify({"ok": True, "stats": stats})
        
    except Exception as e:
        logger.error(f"🔥 Error getting statistics: {str(e)}")
        return jsonify({"ok": False, "stats": None})

@app.route("/model_status", methods=["GET"])
def model_status():
    status = {
        "gemini_llm": GEMINI_API_KEY is not None,
        "gemini_vision": GEMINI_API_KEY is not None,
        "model_name": MODEL_NAME,
        "vision_model_name": VISION_MODEL_NAME,
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024)
    }
    
    return jsonify(status)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    session["conversation_history"] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    session["chat_metadata"] = {
        "start_time": datetime.now().isoformat(),
        "message_count": 0,
        "topics": []
    }
    session.modified = True
    logger.info("🗑️ History cleared")
    return jsonify({"ok": True, "message": "Conversation history cleared"})

@app.errorhandler(404)
def not_found(error):
    return jsonify({"ok": False, "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"ok": False, "message": "Internal server error"}), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"ok": False, "message": "File too large. Maximum size is 16MB."}), 413

# For production servers
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))