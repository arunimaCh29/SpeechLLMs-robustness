# SpeechLLMs-robustness

Speech-based large language models (Speech LLMs) have recently gained significant attention due to their ability to process spoken queries and generate contextually rich responses. They play an important role in conversational AI systems such as virtual assistants, customer support agents, and social robots. While these models demonstrate strong performance on clean audio data, their robustness under real-world noisy conditions is still an open challenge. In many practical deployments, users interact with these systems in environments where background noise, overlapping speech, or recording artifacts can severely degrade the quality of input audio. This lack of robustness can negatively impact accessibility, reliability, and user trust. 

The central goal of this project is to investigate how Speech LLMs handle noisy question-answering tasks, and whether parameter-efficient fine-tuning methods can improve robustness. Specifically, we study quantized low-rank adaptation (QLoRA) improve robustness to noisy input more effectively than training solely on augmented noisy data. Understanding this trade-off is crucial for developing efficient training strategies that are scalable and deployable under realistic compute constraints.

To obtain clean and noisy samples run:

`clean_vs_noise_data.ipynb`

To run the base model for inference use:

`inference.ipynb`

Authors:
- Arunima Chaurasia
- Prachi Sheth
