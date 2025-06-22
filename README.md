mixture of experts implemetation with real and fake dataset 
All images stored in folders: /train/real/, /train/fake/ 
Dataset is sampled to 500 images for each real and fake
Images are resized to (224, 224)
Normalized to [0,1]
Defining models used -
1. Gating funtion : A small neural net that outputs weights over experts for each input. These weights decide how much each expert contributes
2. Expert : A simple dense layer in which each expert processes the same input, independently
3. ForensicMoEBlock : Combines all expert outputs using the gating weights. Uses jnp.sum(gating * expert_outputs) to compute final output
4. VMoEBinaryClassifier : The full model: 2 MoE blocks → Dense → ReLU → Dense(2). Outputs logits for classification (real vs fake)
During validation:
Gets predictions + softmax, Computes classification report and AUC, Logs gating weights for each image, Averages expert weights across all images
Runs for several epochs:
Trains on batches, Evaluates after each epoch
