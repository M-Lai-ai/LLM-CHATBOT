# web_processor.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import re
from pathlib import Path
import hashlib
import time
import html2text
import json
import pdfplumber
import tabula
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI

class WebProcessor:
    def __init__(self, base_url, api_key, output_dir="output", 
                 max_tokens=1000, overlap_tokens=200, 
                 embedding_model="text-embedding-ada-002"):
        """
        Initialise le WebProcessor avec le crawler et le processeur de texte.
        """
        # Configuration de base
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.urls_by_level = {}
        self.all_urls = set()
        self.visited_urls = set()
        
        # Configuration des dossiers
        self.output_dir = output_dir
        self.content_dir = os.path.join(output_dir, "content")
        self.files_dir = os.path.join(output_dir, "files")
        self.embeddings_dir = os.path.join(output_dir, "embeddings")
        self.urls_dir = os.path.join(output_dir, "urls")
        
        # Configuration OpenAI
        self.client = OpenAI(api_key=api_key)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.embedding_model = embedding_model
        self.all_results = []
        
        # Configuration des types de fichiers
        self.file_types = {
            'pdf': ['.pdf'],
            'images': ['.png', '.jpg', '.jpeg', '.gif', '.webp'],
            'word': ['.doc', '.docx'],
            'excel': ['.xls', '.xlsx'],
            'csv': ['.csv'],
            'text': ['.txt']
        }
        
        # Configuration du convertisseur HTML
        self.html_converter = html2text.HTML2Text()
        self.setup_directories()
        self.setup_html_converter()

    def setup_directories(self):
        """Crée tous les dossiers nécessaires"""
        for dir_path in [self.output_dir, self.content_dir, self.files_dir, 
                        self.embeddings_dir, self.urls_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        for file_type in self.file_types:
            folder_path = os.path.join(self.files_dir, file_type)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

    def setup_html_converter(self):
        """Configure les options de conversion HTML vers Markdown"""
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.ignore_emphasis = False
        self.html_converter.ignore_tables = False
        self.html_converter.body_width = 0
        self.html_converter.protect_links = True
        self.html_converter.unicode_snob = True
        self.html_converter.images_to_alt = False
        self.html_converter.default_image_alt = ""

    def is_valid_url(self, url):
        """Vérifie si l'URL appartient au même domaine"""
        return self.domain in url

    def extract_urls_from_page(self, url, current_level):
        """Extrait les URLs d'une page pour un niveau donné"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            new_urls = set()
            
            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    full_url = urljoin(url, href)
                    if self.is_valid_url(full_url) and full_url not in self.all_urls:
                        new_urls.add(full_url)
                        self.all_urls.add(full_url)
            
            if current_level not in self.urls_by_level:
                self.urls_by_level[current_level] = set()
            self.urls_by_level[current_level].update(new_urls)
            
            return new_urls
        except Exception as e:
            print(f"Erreur lors de l'extraction des URLs de {url}: {e}")
            return set()

    def extract_urls_level(self, level):
        """Extrait les URLs pour un niveau spécifique"""
        if level == 1:
            current_urls = {self.base_url}
            self.all_urls.add(self.base_url)
        else:
            if level - 1 not in self.urls_by_level:
                print(f"Le niveau {level-1} n'a pas encore été crawlé")
                return False
            current_urls = self.urls_by_level[level - 1]

        print(f"\nExtraction des URLs de niveau {level}...")
        new_urls = set()
        for url in tqdm(current_urls):
            new_urls.update(self.extract_urls_from_page(url, level))
            time.sleep(0.1)

        print(f"Niveau {level}: {len(new_urls)} nouvelles URLs trouvées")
        return True

    def save_urls_state(self):
        """Sauvegarde l'état des URLs"""
        state = {
            'urls_by_level': {str(k): list(v) for k, v in self.urls_by_level.items()},
            'all_urls': list(self.all_urls),
            'visited_urls': list(self.visited_urls)
        }
        with open(os.path.join(self.urls_dir, 'urls_state.json'), 'w') as f:
            json.dump(state, f, indent=2)

    def load_urls_state(self):
        """Charge l'état des URLs"""
        try:
            with open(os.path.join(self.urls_dir, 'urls_state.json'), 'r') as f:
                state = json.load(f)
                self.urls_by_level = {int(k): set(v) for k, v in state['urls_by_level'].items()}
                self.all_urls = set(state['all_urls'])
                self.visited_urls = set(state['visited_urls'])
                return True
        except FileNotFoundError:
            return False

    def extract_text_from_pdf(self, pdf_path):
        """Extrait le texte et les tableaux d'un PDF"""
        extracted_text = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    extracted_text.append(page.extract_text() or '')
        except Exception as e:
            print(f"Erreur pdfplumber pour {pdf_path}: {e}")

        try:
            tables = tabula.read_pdf(pdf_path, pages='all')
            for i, table in enumerate(tables):
                extracted_text.append(f"\nTableau {i+1}:\n{table.to_string()}\n")
        except Exception as e:
            print(f"Erreur tabula pour {pdf_path}: {e}")

        return "\n".join(extracted_text)

    def process_file(self, url, file_path, file_type):
        """Traite un fichier téléchargé et génère ses embeddings"""
        try:
            if file_type == 'pdf':
                extracted_text = self.extract_text_from_pdf(file_path)
                txt_filename = os.path.splitext(os.path.basename(file_path))[0] + '.txt'
                txt_path = os.path.join(self.content_dir, txt_filename)
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"Source PDF: {url}\n\n---\n\n")
                    f.write(extracted_text)
                
                cleaned_content = self.clean_text(extracted_text)
                chunks = self.split_into_chunks(cleaned_content)
                
                for i, chunk in enumerate(chunks):
                    embedding = self.get_embedding(chunk)
                    if embedding:
                        self.all_results.append({
                            'filename': url,
                            'chunk_id': i,
                            'text': chunk,
                            'embedding': embedding,
                            'type': 'pdf'
                        })
            return True
        except Exception as e:
            print(f"Erreur traitement fichier {url}: {e}")
            return False

    def download_file(self, url):
        """Télécharge et traite les fichiers"""
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                extension = os.path.splitext(url)[1].lower()
                file_type = next((t for t, exts in self.file_types.items() 
                                if extension in exts), None)
                
                if file_type:
                    file_name = hashlib.md5(url.encode()).hexdigest() + extension
                    file_path = os.path.join(self.files_dir, file_type, file_name)
                    
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    if file_type == 'pdf':
                        self.process_file(url, file_path, file_type)
                    
                    return os.path.relpath(file_path, self.content_dir)
        except Exception as e:
            print(f"Erreur téléchargement {url}: {e}")
        return None

    def process_page_content(self, url, content, title=""):
        """Traite le contenu d'une page et génère les embeddings"""
        try:
            full_content = f"# {title}\n\n" if title else ""
            full_content += f"Source: {url}\n\n---\n\n{content}"

            cleaned_content = self.clean_text(full_content)
            chunks = self.split_into_chunks(cleaned_content)

            for i, chunk in enumerate(chunks):
                embedding = self.get_embedding(chunk)
                if embedding:
                    self.all_results.append({
                        'filename': url,
                        'chunk_id': i,
                        'text': chunk,
                        'embedding': embedding,
                        'type': 'webpage'
                    })
            return True
        except Exception as e:
            print(f"Erreur traitement page {url}: {e}")
            return False

    def crawl_url(self, url):
        """Crawle une URL individuelle"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Traiter les fichiers liés
            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    full_url = urljoin(url, href)
                    extension = os.path.splitext(full_url)[1].lower()
                    if extension and any(extension in exts for exts in self.file_types.values()):
                        self.download_file(full_url)
            
            # Traiter le contenu principal
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            if main_content:
                title = soup.title.string.strip() if soup.title else ""
                content = self.html_converter.handle(str(main_content))
                self.process_page_content(url, content, title)
            
            return True
        except Exception as e:
            print(f"Erreur crawling {url}: {e}")
            return False

    def crawl_with_depth(self, max_depth=3):
        """
        Crawle le site avec une profondeur spécifiée
        
        Args:
            max_depth (int): Profondeur maximale du crawling
        """
        print(f"Démarrage du crawling jusqu'au niveau {max_depth}")
        
        # Extraction des URLs par niveau
        for level in range(1, max_depth + 1):
            print(f"\nExtraction niveau {level}")
            self.extract_urls_level(level)
            self.save_urls_state()
        
        # Crawling des URLs collectées
        total_urls = len(self.all_urls)
        print(f"\nCrawling de {total_urls} URLs au total")
        
        for url in tqdm(self.all_urls):
            if url not in self.visited_urls:
                print(f"\nTraitement: {url}")
                if self.crawl_url(url):
                    self.visited_urls.add(url)
                time.sleep(1)
                
                # Sauvegarde périodique
                if len(self.all_results) % 10 == 0:
                    self.save_results()
        
        # Sauvegarde finale
        self.save_results()
        print(f"\nCrawling terminé. {len(self.visited_urls)} pages traitées")

    def save_results(self):
        """Sauvegarde les résultats dans différents formats"""
        try:
            df = pd.DataFrame(self.all_results)
            
            # CSV
            df_csv = df.copy()
            df_csv['embedding'] = df_csv['embedding'].apply(lambda x: ','.join(map(str, x)))
            df_csv.to_csv(os.path.join(self.embeddings_dir, 'embeddings.csv'), index=False)

            # JSON et NPY
            chunks = []
            embeddings_list = []
            
            for _, row in df.iterrows():
                chunks.append({
                    "text": row['text'],
                    "embedding": row['chunk_id'],
                    "metadata": {
                        "filename": row['filename'],
                        "chunk_id": row['chunk_id'],
                        "type": row['type']
                    }
                })
                embeddings_list.append(row['embedding'])

            with open(os.path.join(self.embeddings_dir, 'chunks.json'), 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            np.save(os.path.join(self.embeddings_dir, 'embeddings.npy'), np.array(embeddings_list))
            return True
        except Exception as e:
            print(f"Erreur sauvegarde: {e}")
            return False
