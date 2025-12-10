Déploiement recommandé (Docker → Render / Heroku)

Option rapide et fiable : déployer en utilisant Docker sur Render (ou Heroku). Cela vous permet de contrôler la version Python (3.11) et d'installer TensorFlow.

1) Déployer localement avec Docker (test)

```powershell
# depuis le dossier du projet
docker build -t prediction_pricestock:latest .
docker run -p 8501:8501 prediction_pricestock:latest
# Ouvrir http://localhost:8501
```

2) Déployer sur Render (recommandé)

- Créez un compte sur https://render.com
- Créez un nouveau "Web Service" et connectez votre dépôt GitHub
- Choisissez "Docker" comme méthode (Render détectera le Dockerfile) ou laissez Render builder l'image
- Déployez

3) Déployer sur Heroku (alternative)

```powershell
heroku create my-prediction-app
heroku stack:set container
git push heroku main
```

Notes:
- Si vous préférez continuer avec Streamlit Cloud, il peut être difficile d'installer TensorFlow (wheels non disponibles pour certains Pythons). Docker élimine ce problème.
- Si vous voulez éviter Docker, la meilleure alternative est d'entraîner le modèle localement, sauvegarder les poids (`model.h5`) et reposer l'application sur une version plus légère (scikit-learn) ou héberger le modèle sur un stockage externe et charger uniquement pour l'inférence.
