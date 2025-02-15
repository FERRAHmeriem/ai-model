# ai-model
# Détection de Produits et Analyse de Merchandising

## Présentation
Ce projet vise à détecter les produits en rayon à l’aide d’un modèle d’intelligence artificielle. Le système extrait des informations sur la répartition des produits et compare l’espace occupé par Ramy par rapport à ses concurrents.

## Structure du Projet
- `ai-model.py` : Entraîne et définit le modèle d’IA pour la détection des produits.
- `dataAugmentation.py` : Applique des techniques d’augmentation des données pour améliorer la performance du modèle.
- `objectdetection.py` : Implémente la détection des objets pour identifier et classer les produits.
- `scrap.py` : Scrape des images de produits concurrents pour l'entraînement et l’analyse comparative.
- `testModel.py` : Teste le modèle d’IA sur de nouvelles images.
- `README.md` : Ce fichier de documentation.

## Fonctionnement
1. **Capture d’images** : Prise de photos des rayons en magasin.
2. **Détection des produits** : Identification et classification des produits via le modèle d’IA.
3. **Extraction des données** : Analyse des produits détectés pour mesurer l’espace en rayon.
4. **Comparaison & Insights** : Évaluation de la part de linéaire de Ramy par rapport aux concurrents et recommandations d’optimisation.

## Instructions d’Installation
1. Cloner le dépôt :
   ```bash
   git clone [<repository-url>](https://github.com/FERRAHmeriem/ai-model)
