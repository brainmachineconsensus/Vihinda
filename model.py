import lightgbm as lgb
from utils import predict_yield_per_env

# Assuming 'model' is your trained LightGBM model
model = lgb.Booster(model_file='/kaggle/working/lightgbm_model.txt')

# Exemple d'utilisation : prédire les rendements par environnement pour un hybride spécifique
hybrid_id = 66  # Remplacez par un ID d'hybride valide dans votre dataset
predicted_yields = predict_yield_per_env(hybrid_id, trainer, model)