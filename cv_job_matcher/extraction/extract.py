import json
import PyPDF2
import re
from unidecode import unidecode
import spacy
import ollama




from pdf2image import convert_from_path
import pytesseract

def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image) + "\n"
    return text



nlp = spacy.load('fr_core_news_sm')

def extract_cv_info_by_AI(text):
    print("extract_cv_info_by_tinyllama")

    # Initialize the CV info structure
    cv_info = {
        'name': '',
        'email': '',
        'phone': '',
        'education': [],
        'experience': [],
        'skills': [],
        'langues parlees': [],
    }

    # Create a clear system prompt asking for JSON-structured data in French
    prompt = [
        {
            "role": "system",
            "content": """Extraire les informations du CV au format JSON avec les clés suivantes : 
                        name (nom), email, phone (téléphone), 
                        education (liste), experience (liste), skills (compétences, liste). 
                        Retournez UNIQUEMENT un JSON valide ."""
        },
        {"role": "user", "content": text}
    ]

    # Get the streamed response using the tinyllama model
    ollama_stream = ollama.chat(
        model="deepseek-r1:7b",
        messages=prompt,
        stream=True,
    )

    # Accumulate content from all chunks
    full_response = []
    for chunk in ollama_stream:
        if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
            content = chunk.message.content
            full_response.append(content)
            print("Received chunk:", content)  # Debug output

    # Combine all chunks into a single string
    raw_response = ''.join(full_response).strip()

    # Attempt to parse the response as JSON
    try:
        parsed_data = json.loads(raw_response)
        cv_info.update(parsed_data)
    except json.JSONDecodeError:
        print("Échec de l'analyse du JSON. Tentative d'extraction des données structurées...")
        print("Réponse brute :", raw_response)

        # Extract structured data manually
        cv_info = extract_data_from_text(raw_response, cv_info)

    return cv_info

def extract_cv_info_by_nuextract(text):
    print("extract_cv_info_by_nuextract")

    # Initialisation de la structure d'informations du CV
    cv_info = {
    'name': '',           # Nom de la personne
    'email': '',          # Adresse email
    'phone': '',          # Numéro de téléphone
    'education': [],      # Liste des formations et diplômes
    'Experience': [],     # Liste des expériences professionnelles
    'skills': [],         # Liste des compétences
    'langues parlees': [] # Liste des langues maîtrisées
    }
    
    # Préparation du template et du prompt selon le format attendu par NuExtract
    template = json.dumps(cv_info, indent=0)
    
    prompt = f"""<|input|>
### Template:
{template}

### Text:
{text}

<|output|>"""
    
    # Utilisation d'Ollama pour appeler NuExtract
    response = ollama.generate(
        model="sroecker/nuextract-tiny-v1.5",
        prompt=prompt,
    )
    
    # Récupération de la réponse
    output_text = response['response']
    
    # Tentative d'extraction du JSON valide
    try:
        # Si le modèle retourne directement un JSON valide
        parsed_data = json.loads(output_text)
        cv_info.update(parsed_data)
    except json.JSONDecodeError:
        print("Format JSON non détecté directement, tentative d'extraction...")
        try:
            # Recherche d'un objet JSON dans la sortie
            json_pattern = r'(\{[\s\S]*\})'
            json_match = re.search(json_pattern, output_text)
            if json_match:
                json_str = json_match.group(1)
                parsed_data = json.loads(json_str)
                cv_info.update(parsed_data)
            else:
                print("Aucun JSON valide trouvé dans la sortie.")
        except json.JSONDecodeError:
            print("Échec de l'extraction JSON. Recours à l'extraction manuelle.")
    
    return cv_info




if __name__ == '__main__':
    cv_pdf_path = "/home/lahad/Documents/ALL PROJECT/cv_analyser/data/cv.pdf"
    cv_text = extract_text_with_ocr(cv_pdf_path)
    print("texte extrait du CV:")
    print(cv_text)
    response = extract_cv_info_by_nuextract(cv_text)
    print("Résultats via NuExtract:")
    
    
    print(json.dumps(response, indent=4, ensure_ascii=False))
