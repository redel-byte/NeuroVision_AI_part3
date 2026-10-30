# NeuroVision AI — Part 3 · Détection et localisation de tumeurs cérébrales

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white) ![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-111827) ![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2) ![FastAPI](https://img.shields.io/badge/API-FastAPI-009688) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B) ![Airflow](https://img.shields.io/badge/Orchestration-Airflow-017CEE)

> **Statut :** projet à implémenter (Part 3). Les chemins de modules, commandes d'API et résultats figurant ici décrivent la **structure cible** ; ils ne prétendent pas que le code est déjà livré.
>
> **Usage :** prototype d'apprentissage et de recherche. **Ne pas utiliser pour prendre une décision médicale réelle** sans validation clinique, contrôle de qualité et conformité réglementaire.

## 1. Présentation

Après deux parties consacrées à la **classification d'IRM entières**, NeuroVision AI passe à la **détection d'objets**. L'objectif est de prédire, pour chaque IRM, une ou plusieurs régions suspectes sous la forme :

- d'une **boîte englobante** (`x_min`, `y_min`, `x_max`, `y_max`) ;
- d'une **classe** correspondant à l'annotation du dataset ;
- d'un **score de confiance** associé à la détection.

Le parcours va du contrôle des annotations YOLO jusqu'à l'exposition du modèle par FastAPI, sa visualisation dans Streamlit et l'orchestration par Airflow sur une instance GPU VastAI.

### Objectifs pédagogiques et livrables

| Domaine | Livrable attendu |
| --- | --- |
| Données | Rapport d'audit, images annotées, histogrammes par classe et split sans fuite |
| YOLO | Dataset conforme (`images/`, `labels/`, `data.yaml`) et justification des augmentations |
| Modélisation | Deux runs comparables de YOLOv8 pré-entraîné et checkpoints `best.pt` / `last.pt` |
| Évaluation | Précision, rappel, mAP@0.5, mAP@0.5:0.95, IoU, matrices et erreurs commentées |
| MLflow | Paramètres, métriques, artefacts, version de modèle et alias `champion` |
| Backend | `GET /health`, `POST /predict`, validation des images et Swagger `/docs` |
| Frontend | Application Streamlit consommant uniquement l'API FastAPI |
| MLOps | DAG Airflow avec préparation → entraînement → évaluation → enregistrement/promotion |
| Livraison | README, dépendances figées, tests et planification Jira d'une semaine |

## 2. Stack technique

- **Python**, **uv** pour la gestion des dépendances et le verrouillage des versions (`uv.lock`).
- **Ultralytics YOLOv8** (détection, modèles `yolov8n.pt`, `yolov8s.pt` et éventuellement `yolov8m.pt`), **PyTorch/CUDA**, **OpenCV**.
- **VastAI** pour le GPU (environnement Linux, GPU compatible CUDA).
- **MLflow** Tracking et Model Registry pour comparer les expériences et gérer `champion`.
- **FastAPI** + **Uvicorn** pour l'inférence, **Streamlit** + **requests** pour l'interface.
- **Apache Airflow** pour l'orchestration, **pytest** pour les tests.

> **Note déploiement :** il est préférable d'isoler Airflow dans un environnement/service dédié, compatible avec ses contraintes d'installation, plutôt que de l'ajouter sans vérification à l'environnement d'entraînement GPU.

### Checkpoints et Git LFS

Les checkpoints `.pt` sont versionnés avec Git LFS. Installer Git LFS avant de
cloner le dépôt et récupérer les objets LFS de la branche pour obtenir les vrais
poids, notamment `models/best.pt`. Un petit fichier texte de pointeur LFS ne peut
pas être chargé par YOLO. Les caches de labels, sorties locales répétées,
sauvegardes d'environnement et données de suivi MLflow sont ignorés par Git.

## 3. Dataset

**Nom :** MRI for Brain Tumor with Bounding Boxes  
**Source :** [Kaggle — ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)

**Ne pas supposer** que les dossiers, les étiquettes et les classes correspondent à une organisation spécifique avant le premier audit. Les points à vérifier sont :

1. Présence effective de `Train`, `Valid`, `Test` ou de répertoires équivalents.
2. Format **réel** des annotations : YOLO `.txt` attendu, à confirmer après téléchargement.
3. Index et noms des classes, nombre d'images et nombre de boîtes par classe.
4. Images négatives (sans tumeur), annotations absentes, annotations vides et doublons.
5. Métadonnées patient ou études permettant un découpage **sans fuite par patient**.

### Télécharger sur une machine configurée avec Kaggle

```bash
mkdir -p data/raw
kaggle datasets download \
  -d ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes \
  -p data/raw --unzip
```

Le CLI Kaggle nécessite une authentification préalable. Ne jamais committer de jeton ni de fichier de données confidentielles.

### Comprendre une annotation YOLO

Chaque ligne valide d'un fichier `.txt` pour une boîte de détection comporte :

```text
class_id x_center y_center width height
```

Les quatre coordonnées sont **normalisées dans [0, 1]**. Pour une image de largeur `W` et hauteur `H` :

```text
x_min = (x_center - width / 2) * W
x_max = (x_center + width / 2) * W
y_min = (y_center - height / 2) * H
y_max = (y_center + height / 2) * H
```

**Validation complémentaire :** `class_id` doit être un entier inclus dans `[0, nc-1]`, `width > 0`, `height > 0`, et les bords `(x_center ± width/2, y_center ± height/2)` doivent rester dans `[0, 1]`. Vérifier exactement **5 valeurs** par ligne et **au moins 9 visualisations** avec boîtes et noms de classes.

### Politique de nettoyage

| Situation | Décision proposée |
| --- | --- |
| Image sans annotation | Classer « orpheline » et exclure par défaut, **sauf** si une revue confirme qu'il s'agit d'un négatif : créer alors un `.txt` vide |
| Annotation `.txt` vide | Garder uniquement si l'image est **explicitement confirmée négative** ; documenter le nombre |
| `.txt` sans image | Exclure de la version préparée, compter et journaliser |
| Annotation mal formée / classe inconnue / boîte hors limites | Mettre en quarantaine et consigner la cause ; ne pas corriger silencieusement |
| Extension autre que `.jpeg`, `.jpg`, `.bmp`, `.png` | Écarter et compter |
| Image illisible (OpenCV) | Écarter, journaliser, poursuivre avec `try/except` |
| Image identique/du même patient présente dans plusieurs ensembles | Éliminer la fuite et refaire le split avant tout entraînement |

Les statistiques d'audit doivent contenir le **nombre initial, le nombre rejeté par motif, le nombre conservé**, ainsi que les distributions par classe et par ensemble. Les images et annotations restent appariées à chaque déplacement.

## 4. Structure cible du dépôt

```text
neurovision-ai-part3/
├── README.md
├── pyproject.toml                  # à créer
├── uv.lock                         # généré après résolution
├── .env.example                    # valeurs non secrètes uniquement
├── .gitignore
├── configs/
│   ├── train_baseline.yaml         # hyperparamètres versionnés
│   └── evaluation_policy.yaml      # critère de sélection du champion
├── data/
│   ├── raw/                         # contenu Kaggle, non versionné
│   ├── processed/
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   └── data.yaml
│   └── reports/                     # audit, balance, exemples
├── src/neurovision/
│   ├── data/{inspect_dataset,validate_labels,prepare_dataset}.py
│   ├── training/{train,evaluate,mlflow_tracking}.py
│   ├── registry/{register_model,promote_champion}.py
│   ├── api/{main,schemas,model_service}.py
│   └── common/{config,logging,seed}.py
├── apps/streamlit_app.py
├── dags/neurovision_detection_pipeline.py
├── tests/{test_data,test_api,test_inference}.py
├── artifacts/                      # métriques, figures, rapports
└── runs/                           # checkpoints Ultralytics, non versionné
```

> Les chemins ci-dessus sont **proposés** : créer progressivement les modules pendant le sprint. Un split nommé `Valid` côté source peut être renommé en `val` dans la version préparée, ce qui simplifie `data.yaml`.

### Exemple de `data.yaml` (à générer après l'audit)

```yaml
path: /workspace/neurovision-ai-part3/data/processed  # adapter à VastAI
train: images/train
val: images/val
test: images/test
nc: 1                 # EXEMPLE UNIQUEMENT : remplacer par le nombre réel
names:
  0: tumor           # EXEMPLE UNIQUEMENT : remplacer par les vraies classes
```

Vérifier que les IDs des labels sont alignés sur `names`, et que chaque split contient autant d'images que de fichiers `.txt` associés, en comptant les annotations négatives confirmées. Si aucun split n'existe, créer par exemple **70 % train / 15 % validation / 15 % test** avec graine fixe `42` et stratification raisonnable (images multiboîtes : stratification multilabel ou regroupement par patient si disponible).

## 5. Mise en place de l'environnement

### Python et dépendances

Depuis le dossier du projet, après création de `pyproject.toml` :

```bash
uv sync
```

Pour ajouter les principales dépendances au démarrage :

```bash
uv add ultralytics opencv-python-headless mlflow fastapi \
  'uvicorn[standard]' python-multipart streamlit requests \
  pyyaml pillow kaggle
uv add --dev pytest
uv lock
```

Le projet recommande **Python 3.11** ; vérifier la compatibilité des versions Airflow/PyTorch avant de figer l'environnement. Enregistrer `uv.lock` et la version de CUDA/driver dans le rapport de reproduction.

### Vérification GPU VastAI

```bash
nvidia-smi
uv run python -c 'import torch; print("CUDA disponible:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

Privilégier le GPU pour les entraînements et adapter `batch` à sa mémoire. Si les poids ne peuvent pas être téléchargés depuis l'instance, transférer légalement les poids pré-entraînés vers le serveur et référencer leur chemin local.

## 6. Préparation et augmentation

L'ordre recommandé : **audit → corrections documentées → split → `data.yaml` → contrôles → entraînement**.

Les augmentations ne s'appliquent **qu'à train** :

| Paramètre | Valeur de départ suggérée | Pourquoi |
| --- | --- | --- |
| `degrees` | `5.0` | rotation modérée, anatomie préservée |
| `translate` | `0.05` | robustesse à un léger décalage |
| `scale` | `0.10` | légère variation de taille |
| `hsv_h`, `hsv_s` | `0.0` | pas de changement artificiel de teinte/saturation sur IRM en niveaux de gris |
| `hsv_v` | `0.10` | légère variation de luminosité (à valider visuellement) |
| `fliplr`, `flipud` | `0.0` | éviter de modifier arbitrairement la latéralité/anatomie |
| `mosaic` | `0.0` (point de départ) | éviter de composer plusieurs cerveaux dans une même image |

**Contraste :** examiner une transformation d'intensité/contraste adaptée à l'imagerie médicale si nécessaire ; ne pas confondre `hsv_v` (luminosité) avec un réglage explicite du contraste. Toujours contrôler des exemples augmentés et leurs boîtes transformées. `Mosaic` combine plusieurs images et repositionne/recadre les boîtes ; il peut produire des scènes médicalement peu réalistes.

## 7. Entraînement YOLOv8

**Stratégie de base :** choisir `yolov8n.pt` pour le premier essai (rapide, léger), puis comparer **à paramètres identiques** avec `yolov8s.pt` (plus lourd, souvent meilleur compromis si le GPU le permet). Envisager `yolov8m.pt` seulement si ressources et planning le justifient.

**Test de lecture du dataset, 1 epoch** :

```bash
uv run yolo detect train \
  model=yolov8n.pt data=data/processed/data.yaml \
  epochs=1 imgsz=640 batch=8 device=0 seed=42 \
  project=runs/detect name=smoke_test
```

**Expérience A : baseline** :

```bash
uv run yolo detect train \
  model=yolov8n.pt data=data/processed/data.yaml \
  epochs=50 imgsz=640 batch=8 patience=10 device=0 seed=42 \
  degrees=5 translate=0.05 scale=0.10 \
  fliplr=0 flipud=0 hsv_h=0 hsv_s=0 hsv_v=0.10 mosaic=0.0 \
  project=runs/detect name=baseline_n
```

**Expérience B : changement d'un seul paramètre (`model`)** :

```bash
uv run yolo detect train \
  model=yolov8s.pt data=data/processed/data.yaml \
  epochs=50 imgsz=640 batch=8 patience=10 device=0 seed=42 \
  degrees=5 translate=0.05 scale=0.10 \
  fliplr=0 flipud=0 hsv_h=0 hsv_s=0 hsv_v=0.10 mosaic=0.0 \
  project=runs/detect name=baseline_s
```

Ces commandes sont des **configurations initiales**, non des hyperparamètres optimaux prouvés. En cas d'OOM, réduire le batch pour **les deux runs** afin de préserver la comparaison. Chronométrer avec `time.perf_counter()` autour de `model.train()` dans les scripts de suivi MLflow.

Sorties attendues selon la version/configuration Ultralytics : `weights/best.pt`, `weights/last.pt`, `results.csv`, courbes, visualisations et matrice de confusion dans `runs/detect/<nom_du_run>/`.

**Reprise d'un entraînement interrompu :**

```bash
uv run yolo detect train model=runs/detect/baseline_n/weights/last.pt resume=True
```

Observer les pertes de boîtes, de classification (et DFL selon la version). Des pertes d'entraînement qui baissent sans amélioration de validation peuvent indiquer un **surapprentissage** ; utiliser early stopping (`patience`) et revoir les augmentations.

## 8. Évaluation et analyse d'erreurs

1. Comparer **sur validation** la précision, le rappel, `mAP@0.5` et `mAP@0.5:0.95`, globalement **et par classe**.
2. Inspecter courbes précision–rappel et F1, matrices de confusion et boîtes mal localisées.
3. Afficher exemples de vrais positifs, faux positifs et faux négatifs **avec annotations de référence et prédictions**.
4. Calculer des IoU d'exemples : `IoU = aire(intersection) / aire(union)`, sans confondre IoU et score de confiance.
5. Fixer **avant le test** la règle de sélection du modèle selon le **rappel élevé** et un niveau acceptable de mAP/FP ; consigner les seuils dans `configs/evaluation_policy.yaml`.
6. Sélectionner sur validation et utiliser **le jeu de test une seule fois** pour le bilan final ; ne pas réajuster les paramètres/thresholds d'après ce test.

**Évaluation test, uniquement après sélection finale** :

```bash
uv run yolo detect val \
  model=runs/detect/baseline_n/weights/best.pt \
  data=data/processed/data.yaml split=test imgsz=640 device=0
```

Remplacer `baseline_n` par le run réellement sélectionné. Pour les futures itérations, conserver le jeu test scellé ; la promotion automatique du champion doit se baser sur **validation** ou sur un banc d'évaluation dédié, pas sur le test final réutilisé.

## 9. MLflow : traçabilité et sélection du `champion`

Le suivi manuel du notebook est implémenté dans
`src/neurovision_ai_part3/mlflow_tracking.py` : fonctions simples pour les
paramètres, métriques, historiques et artefacts. Voir le
[guide de démarrage MLflow](docs/MLFLOW.md) pour Windows, Vast.ai et la vérification
dans l'interface. Le packaging Registry et l'alias `champion` restent à construire.

### Démarrer le serveur local (exemple développement)

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --serve-artifacts --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 --port 5000
```

Ouvrir <http://127.0.0.1:5000>. Le serveur de suivi doit être accessible depuis les processus d'entraînement, mais **ne doit pas être exposé sans authentification** sur un port public VastAI.

Variables d'environnement à utiliser dans les scripts futurs :

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=NeuroVision_AI_Part3_Detection
export MODEL_NAME=NeuroVisionBrainTumorDetector
```

Pour **chaque run**, enregistrer : `model`, `seed`, `imgsz`, `batch`, `epochs`, `patience`, augmentations, hash/version du dataset, versions d'outils, GPU ; `precision`, `recall`, `mAP50`, `mAP50-95`, métriques par classe ; durée d'entraînement ; `best.pt`, `data.yaml`, `results.csv`, figures et rapport d'erreurs`.

### Model Registry

Enregistrer un **véritable modèle MLflow** (par exemple via un wrapper `mlflow.pyfunc` qui charge le `.pt` YOLO et convertit les résultats) dans `NeuroVisionBrainTumorDetector`. **Un fichier `.pt` journalisé comme simple artefact n'est pas automatiquement une version enregistrée exploitable comme modèle MLflow.**

L'alias `champion` désigne la version autorisée pour l'API :

```python
from mlflow import MlflowClient

client = MlflowClient()
selected = client.get_model_version_by_alias(
    name="NeuroVisionBrainTumorDetector", alias="champion"
)
print(selected.version)

# Réaffecter seulement après validation de la politique de promotion :
# client.set_registered_model_alias("NeuroVisionBrainTumorDetector", "champion", new_version)
```

La règle de promotion doit comparer des métriques **de validation** comparables et garder la version actuelle si le candidat ne franchit pas les seuils définis. Enregistrer la décision, les métriques et l'ancien/nouveau numéro de version.

## 10. API FastAPI (implémentation locale)

L'API dans `src/neurovision_ai_part3/api.py` charge **une seule fois au démarrage**
le checkpoint local `models/best.pt`. Elle retourne son empreinte SHA-256 et les
classes lues dans le modèle. Le chargement d'un modèle Registry via
`models:/NeuroVisionBrainTumorDetector@champion` reste une étape future, après
validation de la promotion. Voir le [guide API](docs/API.md) pour les requêtes,
la configuration et les tests.

| Méthode | Route | Comportement attendu |
| --- | --- | --- |
| GET | `/health` | Retourne l'état du modèle, le device, le checkpoint, son hash et les classes ; 503 si indisponible |
| POST | `/predict` | `multipart/form-data` avec `file`, seuil optionnel `confidence` entre 0 et 1 ; retourne les détections en JSON |
| GET | `/docs` | Documentation Swagger fournie par FastAPI |

Règles d'entrée : types autorisés `.jpeg`, `.jpg`, `.bmp`, `.png` **et** contenu réellement décodable ; limites configurables (10 Mio et 16 millions de pixels par défaut) ; `400` pour extension non prise en charge, `413` pour limite dépassée, `422` pour fichier manquant/illisible ou paramètre invalide, `503` si le modèle ne peut pas être chargé.

**Exemple de réponse illustratif (valeurs fictives)** :

```json
{
  "message": "Detected 1 region(s).",
  "model": {
    "source": "local_checkpoint",
    "checkpoint": "best.pt",
    "sha256": "<empreinte SHA-256 du checkpoint>",
    "classes": {"0": "Glioma", "1": "Meningioma", "2": "Pituitary"}
  },
  "image_width": 512,
  "image_height": 512,
  "confidence_threshold": 0.25,
  "detections": [
    {
      "class_id": 0,
      "class_name": "Glioma",
      "confidence": 0.91,
      "bbox_xyxy": [72, 44, 211, 188]
    }
  ]
}
```

Sans détection, la réponse conserve les métadonnées et contient
`"message": "No detections at this confidence threshold."` et `"detections": []`.
Cela signifie **aucune détection au seuil choisi**, et **non** l'absence cliniquement prouvée de tumeur.

**Démarrer depuis la racine du dépôt, dans PowerShell** :

```powershell
.\.venv\Scripts\python.exe -m uvicorn neurovision_ai_part3.api:app --app-dir src --host 127.0.0.1 --port 8000
```

**Dans un second terminal** (remplacer le chemin de l'image) :

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST "http://127.0.0.1:8000/predict?confidence=0.25" -F "file=@C:\path\to\mri.jpg"
```

Accéder à la documentation sur <http://127.0.0.1:8000/docs>.

## 11. Interface Streamlit (contrat cible)

L'interface n'instancie **jamais** YOLO ni MLflow en direct : elle transmet l'image à `POST /predict` et visualise les boîtes renvoyées par FastAPI.

Fonctions à construire : import d'une IRM, curseur de seuil de confiance, overlay boîtes/classe/score, message « aucune tumeur détectée », états d'erreur (API inaccessible, fichier vide, mauvais format, image illisible), délai raisonnable pour la requête.

**Après implémentation de `apps/streamlit_app.py`** :

```bash
export NEUROVISION_API_URL=http://127.0.0.1:8000
uv run streamlit run apps/streamlit_app.py
```

Ne pas stocker d'IRM utilisateurs ni publier d'images sensibles par défaut ; logs minimaux et suppression explicite des fichiers temporaires.

## 12. Airflow : pipeline automatisé

**DAG cible :** `neurovision_detection_pipeline` exécuté sur l'instance VastAI (ou depuis un orchestrateur ayant accès à ses ressources).

```text
prepare_dataset
       │
       ▼
train_yolov8
       │
       ▼
evaluate_on_validation
       │
       ▼
register_candidate
       │
       ▼
compare_and_promote_champion
```

- Exécution conditionnée à la réussite de la tâche précédente ; erreurs remontées dans les logs, `retries` et délai d'attente configurés.
- Tous les artefacts nécessaires sont **persistants** : disque de l'instance ou stockage distant ; éviter de tout perdre quand l'instance VastAI est arrêtée.
- Pour le sprint : **déclenchement manuel** au début, puis calendrier explicite (par exemple `@weekly`) une fois le pipeline fiable.
- La promotion doit être **atomique et idempotente** autant que possible : aucun déplacement de l'alias `champion` si l'évaluation échoue, est indisponible ou n'améliore pas le critère validé.
- Le DAG d'entraînement/validation **n'exécute pas systématiquement** un nouveau test final, qui resterait réservé au bilan convenu.

L'installation d'Airflow doit suivre la documentation officielle de la version choisie et utiliser ses contraintes de dépendances.

## 13. Tests, sécurité et reproductibilité

**Tests minimaux attendus** :

- **Data :** parsing YOLO, boîte aux limites, `class_id` invalide, orphelins, `.txt` vide confirmé négatif, image corrompue, absence de fuite entre splits.
- **API :** `/health`, image autorisée, mauvais format, image illisible, `confidence` hors `[0, 1]`, absence de détections, modèle indisponible.
- **UI :** résultat avec/sans boîtes et gestion de panne de l'API.
- **Airflow :** enchaînement des tâches, retry, échec contrôlé, décision de promotion avec/sans amélioration.
- **Reproductibilité :** `seed=42`, versions figées, configuration du GPU/CUDA, hash du dataset, configs, commandes et métriques de chaque run.

**Sécurité :** valider la taille et le contenu des fichiers envoyés, ne pas exposer MLflow/Airflow sans sécurité réseau/authentification, ne pas committer les secrets Kaggle/MLflow ni des images médicales sensibles, ne pas prétendre à une fiabilité clinique sur la seule base du mAP.

## 14. Planification Jira — sprint de 7 jours

**Période : lundi 26 octobre → dimanche 1er novembre 2026.** Travail séquencé en **9 Epics et leurs tâches**, sans extension au-delà d'une semaine. C'est un calendrier **accéléré de prototype pédagogique**, dépendant de l'accès GPU et du temps d'entraînement.

| Jour | Date | Objectif / jalon |
| --- | --- | --- |
| J1 | Lun. 26/10 | Audit complet ; données nettoyées ; split et `data.yaml` |
| J2 | Mar. 27/10 | Augmentations ; GPU ; MLflow ; smoke test ; démarrage des deux runs |
| J3 | Mer. 28/10 | Comparaison en validation, analyse des erreurs, test final unique et Registry `champion` |
| J4 | Jeu. 29/10 | FastAPI : chargement du champion, endpoints, tests et Swagger |
| J5 | Ven. 30/10 | Streamlit, visualisations et test d'intégration UI ↔ API |
| J6 | Sam. 31/10 | Airflow : DAG, retries, promotion conditionnelle, exécution supervisée |
| J7 | Dim. 01/11 | QA finale, reproductibilité, documentation et démonstration |

**Ordre bloquant :** validation du dataset → entraînement + suivi MLflow → choix sur validation → enregistrement → API → Streamlit → DAG intégrateur → démo. Le choix du champion doit être versionné avant que l'API soit utilisée.

### Import du fichier Jira

Fichier livré séparément : `NeuroVision_AI_Part3_Jira_1week.csv`.

1. Dans **Jira Cloud**, utiliser l'**import CSV via l'import d'un système externe** / import avec hiérarchie, **pas** la création CSV simplifiée sans hiérarchie.
2. Choisir votre **projet Jira existant** comme cible ; aucun identifiant de projet ni utilisateur n'est inventé dans le fichier.
3. Mapper exactement les en-têtes CSV aux champs Jira :

   | En-tête CSV | Champ Jira à sélectionner |
   | --- | --- |
   | `Summary` | Summary |
   | `Description` | Description |
   | `Work type` | Work type / Issue Type |
   | `Work item ID` | Work item ID / Issue ID |
   | `Parent` | Parent |
   | `Status` | Status |
   | `Priority` | Priority |
   | `Due date` | Due date |
   | `Labels` | Labels |

4. Mapper les types `Epic` et `Task` vers les types existant dans votre projet, et l'état `To Do` vers son état équivalent.
5. Pour **Due date**, préciser **`yyyy-MM-dd`** lors de l'import ; chaque date a le format `2026-10-26`.
6. **Garder l'ordre des lignes** : tous les Epics précèdent leurs tâches ; `Parent` contient l'**identifiant numérique `Work item ID` de l'Epic**, **pas** une clé Jira `PROJ-123`.
7. Vérifier après l'import qu'au moins une tâche apparaît sous chaque Epic, puis créer votre **sprint de 7 jours** et y placer les tâches. L'import CSV ne garantit pas la création automatique d'un sprint.

Les exigences de mapping et de hiérarchie correspondent aux ressources officielles Atlassian : [préparation CSV](https://support.atlassian.com/jira-software-cloud/docs/prepare-a-csv-file-for-import/) et [import avec Parent](https://support.atlassian.com/jira-cloud-administration/docs/import-data-from-a-csv-file/).

## 15. Résultats — à compléter après exécution

| Modèle / run | Rappel val. | Précision val. | mAP@0.5 val. | mAP@0.5:0.95 val. | Durée | Décision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| YOLOv8n / baseline_n | À mesurer | À mesurer | À mesurer | À mesurer | À mesurer | En attente |
| YOLOv8s / baseline_s | À mesurer | À mesurer | À mesurer | À mesurer | À mesurer | En attente |

**Bilan test unique :** à renseigner après la sélection du champion (métriques globales/par classe, exemples d'erreurs, limites, version du modèle et hash du dataset).

## 16. Références

- [Dataset Kaggle](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)
- [Ultralytics : modèles YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [Ultralytics : entraînement](https://docs.ultralytics.com/modes/train/)
- [Ultralytics : validation](https://docs.ultralytics.com/modes/val/)
- [MLflow : Model Registry](https://mlflow.org/docs/latest/model-registry/)
- [FastAPI : UploadFile](https://fastapi.tiangolo.com/tutorial/request-files/)
- [Streamlit : file uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
- [Apache Airflow : documentation](https://airflow.apache.org/docs/)
- [Atlassian : importer un CSV dans Jira](https://support.atlassian.com/jira-cloud-administration/docs/import-data-from-a-csv-file/)

---

**NeuroVision AI / Part 3** — Projet pédagogique de computer vision, détection d'objets et MLOps.
