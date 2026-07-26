import json

EVAL_SET = [
    {
        "question": "What is the total number of samples in the combined dataset?",
        "ground_truth": "16,065",
    },
    {
        "question": "How many samples come from the field (PlantDoc) domain?",
        "ground_truth": "1,134",
    },
    {
        "question": "What backbone architecture is used for the visual encoding stream?",
        "ground_truth": "Swin-Tiny Transformer",
    },
    {
        "question": "What patch size does the Swin-Tiny Transformer use to divide input images?",
        "ground_truth": "4x4 patches",
    },
    {
        "question": "What absolute accuracy gain did meteorological integration provide over vision-only models?",
        "ground_truth": "7.29%",
    },
    {
        "question": "What accuracy does the model achieve when weather data is zeroed out (Experiment A1)?",
        "ground_truth": "61.76%",
    },
    {
        "question": "What accuracy does the model achieve in the Scrambled weather ablation experiment (A2)?",
        "ground_truth": "65.69%",
    },
    {
        "question": "What accuracy does the Weather-Only ablation experiment (A3) achieve?",
        "ground_truth": "68.00%",
    },
    {
        "question": "What accuracy does the proposed Swin model achieve on the held-out PlantDoc test set?",
        "ground_truth": "85.29%",
    },
    {
        "question": "What is the mAP of the proposed Swin model?",
        "ground_truth": "0.9195",
    },
    {
        "question": "What explainability technique is used to confirm the model attends to diagnostic lesions?",
        "ground_truth": "LIME (Local Interpretable Model-agnostic Explanations)",
    },
    {
        "question": "What relative humidity threshold is associated with Late Blight in the epidemiological parameters table?",
        "ground_truth": ">90%",
    },
    {
        "question": "What is a stated limitation regarding the meteorological data used in this study?",
        "ground_truth": "The meteorological vectors are synthetic, not real-world sensor data.",
    },
    {
        "question": "How many images are in the test set mentioned in the limitations section?",
        "ground_truth": "102 images",
    },
    {
        "question": "What percentage of Mosaic Virus samples are misclassified as Healthy, and why?",
        "ground_truth": "30%, due to the effect of sunlight glare.",
    },
]

if __name__ == "__main__":
    with open("data/eval_set.json", "w", encoding="utf-8") as f:
        json.dump(EVAL_SET, f, indent=2)

    print(f"Wrote {len(EVAL_SET)} Q&A pairs to data/eval_set.json")
