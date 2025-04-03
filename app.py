import os
import re
import pandas as pd
import numpy as np
import nltk
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader
import docx
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy model for NER
try:
    nlp = spacy.load('en_core_web_sm')
except:
    logger.info("Downloading spaCy model")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class ResumeParser:
    """Parse resumes in various formats and extract text"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
        
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF files"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX files"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, file_path):
        """Extract text from TXT files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {str(e)}")
            return ""
    
    def parse_resume(self, file_path):
        """Parse resume based on file extension"""
        _, file_extension = os.path.splitext(file_path.lower())
        
        if file_extension not in self.supported_formats:
            logger.warning(f"Unsupported file format: {file_extension}")
            return None
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return self.extract_text_from_txt(file_path)


class ResumeInformationExtractor:
    """Extract key information from resume text"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Patterns for information extraction
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_pattern = r'(\+\d{1,3}[-\.\s]??)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}'
        
        # Keywords for sections
        self.education_keywords = ['education', 'academic', 'university', 'college', 'degree', 'bachelor', 'master', 'phd']
        self.experience_keywords = ['experience', 'employment', 'work', 'job', 'career', 'professional']
        self.skills_keywords = ['skills', 'abilities', 'competencies', 'proficiencies', 'expertise']
        
        # Technical skills list
        self.tech_skills = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'rust',
            'sql', 'nosql', 'mongodb', 'mysql', 'postgresql', 'oracle', 'sqlite',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'asp.net',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'terraform',
            'machine learning', 'deep learning', 'neural networks', 'natural language processing', 
            'computer vision', 'data science', 'data analysis', 'big data', 'hadoop', 'spark',
            'html', 'css', 'bootstrap', 'jquery', 'rest api', 'graphql'
        ]
    
    def preprocess_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return " ".join(processed_tokens)
    
    def extract_contact_info(self, text):
        """Extract email and phone number"""
        email = re.search(self.email_pattern, text)
        phone = re.search(self.phone_pattern, text)
        
        contact_info = {
            'email': email.group(0) if email else None,
            'phone': phone.group(0) if phone else None
        }
        
        return contact_info
    
    def extract_education(self, text):
        """Extract education information using NER"""
        education_info = []
        doc = nlp(text)
        
        # Find education section
        text_lower = text.lower()
        education_section = None
        
        for keyword in self.education_keywords:
            if keyword in text_lower:
                # Find the approximate section (primitive approach)
                pattern = re.compile(f".*{keyword}.*", re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    start_idx = match.start()
                    # Get text chunk after education keyword
                    education_section = text[start_idx:start_idx + 1000]  # Limit to 1000 chars
                    break
        
        if education_section:
            # Look for educational institutions and degrees
            doc = nlp(education_section)
            
            # Extract organizations (potential universities) and dates
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
            
            # Extract degree keywords
            degree_keywords = ["bachelor", "master", "phd", "doctorate", "bs", "ms", "ba", "ma", "mba"]
            degrees = []
            
            for keyword in degree_keywords:
                if keyword in education_section.lower():
                    # Find the full degree mention
                    pattern = re.compile(f"[^.]*{keyword}[^.]*", re.IGNORECASE)
                    matches = pattern.findall(education_section)
                    degrees.extend(matches)
            
            # Combine information
            for i, org in enumerate(orgs):
                edu_entry = {"institution": org}
                
                # Try to match a degree with this institution
                if i < len(degrees):
                    edu_entry["degree"] = degrees[i].strip()
                
                # Try to match dates
                if i < len(dates):
                    edu_entry["date"] = dates[i]
                    
                education_info.append(edu_entry)
                
        return education_info
    
    def extract_experience(self, text):
        """Extract work experience information"""
        experience_info = []
        doc = nlp(text)
        
        # Find experience section
        text_lower = text.lower()
        experience_section = None
        
        for keyword in self.experience_keywords:
            if keyword in text_lower:
                # Find the approximate section
                pattern = re.compile(f".*{keyword}.*", re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    start_idx = match.start()
                    # Get text chunk after experience keyword
                    experience_section = text[start_idx:start_idx + 2000]  # Limit to 2000 chars
                    break
        
        if experience_section:
            # Extract organizations and dates
            doc = nlp(experience_section)
            
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
            
            # Try to extract job titles (this is approximate)
            job_title_patterns = [
                r'(Software Engineer|Developer|Programmer|Analyst|Manager|Director|Consultant|Architect|Administrator|Designer|Lead|Head)'
            ]
            
            job_titles = []
            for pattern in job_title_patterns:
                matches = re.findall(pattern, experience_section)
                job_titles.extend(matches)
            
            # Combine information
            for i, org in enumerate(orgs):
                exp_entry = {"company": org}
                
                # Try to match a job title
                if i < len(job_titles):
                    exp_entry["title"] = job_titles[i]
                
                # Try to match dates
                if i < len(dates):
                    exp_entry["date"] = dates[i]
                    
                experience_info.append(exp_entry)
        
        return experience_info
    
    def extract_skills(self, text):
        """Extract technical skills from resume"""
        skills_found = []
        text_lower = text.lower()
        
        for skill in self.tech_skills:
            if skill in text_lower:
                skills_found.append(skill)
        
        # Try to find skills section for more context
        skills_section = None
        for keyword in self.skills_keywords:
            if keyword in text_lower:
                pattern = re.compile(f".*{keyword}.*", re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    start_idx = match.start()
                    skills_section = text[start_idx:start_idx + 1000]
                    break
        
        if skills_section:
            # Analyze skills section for additional skills
            doc = nlp(skills_section)
            
            # Extract nouns that might be skills not in our predefined list
            potential_skills = [token.text.lower() for token in doc if token.pos_ == "NOUN" 
                               and token.text.lower() not in self.stop_words
                               and len(token.text) > 2]
            
            # Add unique skills
            for skill in potential_skills:
                if skill not in skills_found:
                    skills_found.append(skill)
        
        return skills_found
    
    def extract_all_information(self, resume_text):
        """Extract all relevant information from resume text"""
        if not resume_text:
            return None
            
        contact_info = self.extract_contact_info(resume_text)
        education = self.extract_education(resume_text)
        experience = self.extract_experience(resume_text)
        skills = self.extract_skills(resume_text)
        
        # Create full resume information dictionary
        resume_info = {
            'contact_info': contact_info,
            'education': education,
            'experience': experience,
            'skills': skills,
            'raw_text': resume_text,
            'processed_text': self.preprocess_text(resume_text)
        }
        
        return resume_info


class ResumeRanker:
    """Rank resumes based on job requirements using TF-IDF and cosine similarity"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
    def fit_vectorizer(self, resume_texts):
        """Fit the TF-IDF vectorizer on resume texts"""
        if not resume_texts:
            logger.warning("No resume texts provided for fitting vectorizer")
            return None
        
        self.vectorizer.fit(resume_texts)
        return self.vectorizer
    
    def rank_resumes(self, resume_data, job_description, weight_skills=0.5, weight_experience=0.3, weight_education=0.2):
        """
        Rank resumes based on similarity to job description
        
        Args:
            resume_data: List of dictionaries containing processed resume information
            job_description: The job description text
            weight_skills: Weight for skills score (default: 0.5)
            weight_experience: Weight for experience score (default: 0.3)
            weight_education: Weight for education score (default: 0.2)
            
        Returns:
            DataFrame with ranked resumes
        """
        if not resume_data or not job_description:
            logger.warning("Missing data for ranking resumes")
            return None
            
        # Extract processed texts
        resume_texts = [resume['processed_text'] for resume in resume_data if 'processed_text' in resume]
        
        if not resume_texts:
            logger.warning("No processed resume texts available for ranking")
            return None
            
        # Preprocess job description
        extractor = ResumeInformationExtractor()
        processed_job_desc = extractor.preprocess_text(job_description)
        
        # Extract skills from job description
        job_skills = extractor.extract_skills(job_description)
        
        # Fit vectorizer if not already fit
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.fit_vectorizer(resume_texts + [processed_job_desc])
        
        # Transform resume texts and job description
        resume_vectors = self.vectorizer.transform(resume_texts)
        job_vector = self.vectorizer.transform([processed_job_desc])
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(resume_vectors, job_vector).flatten()
        
        # Calculate additional scores for each resume
        skill_scores = []
        experience_scores = []
        education_scores = []
        
        for resume in resume_data:
            # Skills score - match skills with job requirements
            resume_skills = resume.get('skills', [])
            if resume_skills and job_skills:
                skill_match = len(set(resume_skills).intersection(set(job_skills))) / len(job_skills) if job_skills else 0
                skill_scores.append(skill_match)
            else:
                skill_scores.append(0)
            
            # Experience score - based on years and relevance
            experience = resume.get('experience', [])
            experience_score = min(len(experience) * 0.2, 1.0)  # Scale based on number of positions, max 1.0
            experience_scores.append(experience_score)
            
            # Education score - simple presence check
            education = resume.get('education', [])
            education_score = min(len(education) * 0.5, 1.0)  # Scale based on number of degrees, max 1.0
            education_scores.append(education_score)
        
        # Calculate weighted scores
        weighted_scores = (
            weight_skills * np.array(skill_scores) +
            weight_experience * np.array(experience_scores) +
            weight_education * np.array(education_scores) +
            (1 - weight_skills - weight_experience - weight_education) * cosine_similarities
        )
        
        # Create rankings dataframe
        rankings = []
        for i, resume in enumerate(resume_data):
            email = resume.get('contact_info', {}).get('email', 'Unknown')
            skills_list = resume.get('skills', [])
            
            rankings.append({
                'rank': i+1,  # Will be updated after sorting
                'email': email,
                'similarity_score': cosine_similarities[i],
                'skill_score': skill_scores[i],
                'experience_score': experience_scores[i],
                'education_score': education_scores[i],
                'weighted_score': weighted_scores[i],
                'matched_skills': ', '.join(set(skills_list).intersection(set(job_skills))),
                'missing_skills': ', '.join(set(job_skills) - set(skills_list))
            })
        
        # Create DataFrame and sort by weighted score
        rankings_df = pd.DataFrame(rankings)
        rankings_df = rankings_df.sort_values(by='weighted_score', ascending=False).reset_index(drop=True)
        
        # Update ranks
        rankings_df['rank'] = rankings_df.index + 1
        
        return rankings_df


class ResumeScreener:
    """Main class for resume screening pipeline"""
    
    def __init__(self, resume_dir=None):
        self.parser = ResumeParser()
        self.extractor = ResumeInformationExtractor()
        self.ranker = ResumeRanker()
        self.resume_dir = resume_dir
        
    def set_resume_directory(self, directory):
        """Set or change resume directory"""
        if os.path.isdir(directory):
            self.resume_dir = directory
            return True
        else:
            logger.error(f"Invalid directory: {directory}")
            return False
    
    def process_resumes(self):
        """Process all resumes in the directory"""
        if not self.resume_dir:
            logger.error("Resume directory not set")
            return None
            
        resume_data = []
        
        for filename in os.listdir(self.resume_dir):
            file_path = os.path.join(self.resume_dir, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
                
            # Parse resume
            resume_text = self.parser.parse_resume(file_path)
            
            if resume_text:
                # Extract information
                resume_info = self.extractor.extract_all_information(resume_text)
                
                if resume_info:
                    resume_info['filename'] = filename
                    resume_data.append(resume_info)
                    logger.info(f"Successfully processed: {filename}")
                else:
                    logger.warning(f"Failed to extract information from: {filename}")
            else:
                logger.warning(f"Failed to parse: {filename}")
        
        logger.info(f"Processed {len(resume_data)} resumes")
        return resume_data
    
    def screen_resumes(self, job_description, custom_weights=None):
        """Screen resumes against a job description"""
        # Process resumes if not already done
        resume_data = self.process_resumes()
        
        if not resume_data:
            logger.error("No resume data available for screening")
            return None
        
        # Set custom weights if provided
        if custom_weights:
            weight_skills = custom_weights.get('skills', 0.5)
            weight_experience = custom_weights.get('experience', 0.3)
            weight_education = custom_weights.get('education', 0.2)
        else:
            weight_skills = 0.5
            weight_experience = 0.3
            weight_education = 0.2
        
        # Rank resumes
        rankings = self.ranker.rank_resumes(
            resume_data, 
            job_description,
            weight_skills=weight_skills,
            weight_experience=weight_experience,
            weight_education=weight_education
        )
        
        return rankings
    
    def save_results(self, rankings, output_file='resume_rankings.csv'):
        """Save ranking results to CSV"""
        if rankings is not None:
            rankings.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            return True
        else:
            logger.error("No rankings to save")
            return False
    
    def save_model(self, filename='resume_screener.pkl'):
        """Save the model for later use"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'ranker_vectorizer': self.ranker.vectorizer
                }, f)
            logger.info(f"Model saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filename='resume_screener.pkl'):
        """Load a previously saved model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
                self.ranker.vectorizer = model_data['ranker_vectorizer']
            logger.info(f"Model loaded from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Example job description
    job_description = """
    We are looking for a skilled Python Developer to join our engineering team.
    
    Requirements:
    - Proficient in Python, Django, and Flask
    - Experience with SQL and NoSQL databases
    - Knowledge of RESTful APIs and GraphQL
    - Familiarity with AWS services
    - Experience with Docker and Kubernetes
    - Bachelor's degree in Computer Science or related field
    - 3+ years of experience in software development
    """
    
    # Initialize the resume screener
    screener = ResumeScreener("./resumes")
    
    # Process and rank resumes
    rankings = screener.screen_resumes(job_description)
    
    # Save results
    if rankings is not None:
        screener.save_results(rankings)
        
        # Display top 5 candidates
        print("\nTop 5 Candidates:")
        print(rankings.head(5)[['rank', 'email', 'weighted_score', 'matched_skills']])
        
        # Save the model
        screener.save_model()