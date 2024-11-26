# text_processor.py
import os
import re
import tiktoken
from openai import OpenAI
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

class TextProcessor:
    def __init__(self, 
                 api_key, 
                 folder_path=None,
                 max_tokens=1000,
                 overlap_tokens=200,
                 embedding_model="text-embedding-ada-002"):
        """
        Initialise le TextProcessor avec les paramètres configurables.
        """
        self.client = OpenAI(api_key=api_key)
        self.folder_path = folder_path
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.embedding_model = embedding_model
        self.all_results = []

    # ... [Les méthodes précédentes restent identiques jusqu'à save_results] ...

    def save_results(self, output_format='all', output_prefix='embeddings_results'):
        """
        Sauvegarde les résultats dans différents formats.
        
        Args:
            output_format (str): Format de sortie ('csv', 'json_npy', ou 'all')
            output_prefix (str): Préfixe pour les noms de fichiers de sortie
        """
        try:
            # Créer le DataFrame
            df = pd.DataFrame(self.all_results)
            
            if output_format in ['csv', 'all']:
                # Sauvegarde CSV
                df_csv = df.copy()
                df_csv['embedding'] = df_csv['embedding'].apply(lambda x: ','.join(map(str, x)))
                csv_path = f'{output_prefix}.csv'
                df_csv.to_csv(csv_path, index=False)
                print(f"Résultats sauvegardés dans {csv_path}")

            if output_format in ['json_npy', 'all']:
                # Préparation des chunks pour JSON
                chunks = []
                embeddings_list = []
                
                for _, row in df.iterrows():
                    # Préparer le chunk pour JSON
                    chunk = {
                        "text": row['text'],
                        "embedding": row['chunk_id'],  # Index d'embedding
                        "metadata": {
                            "filename": row['filename'],
                            "chunk_id": row['chunk_id']
                        }
                    }
                    chunks.append(chunk)
                    
                    # Préparer l'embedding pour NPY
                    embeddings_list.append(row['embedding'])

                # Sauvegarder JSON
                json_path = f'{output_prefix}_chunks.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                print(f"Chunks sauvegardés dans {json_path}")

                # Sauvegarder NPY
                embeddings_array = np.array(embeddings_list)
                npy_path = f'{output_prefix}_embeddings.npy'
                np.save(npy_path, embeddings_array)
                print(f"Embeddings sauvegardés dans {npy_path}")

            return True

        except Exception as e:
            print(f"Erreur lors de la sauvegarde des résultats: {e}")
            return False

    def load_results(self, input_prefix='embeddings_results'):
        """
        Charge les résultats à partir des fichiers sauvegardés.
        
        Args:
            input_prefix (str): Préfixe des fichiers à charger
            
        Returns:
            tuple: (chunks, embeddings) si le chargement réussit, None sinon
        """
        try:
            # Charger les chunks depuis JSON
            json_path = f'{input_prefix}_chunks.json'
            with open(json_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            # Charger les embeddings depuis NPY
            npy_path = f'{input_prefix}_embeddings.npy'
            embeddings = np.load(npy_path)

            return chunks, embeddings

        except Exception as e:
            print(f"Erreur lors du chargement des résultats: {e}")
            return None
