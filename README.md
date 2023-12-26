# Prior Preserved Concept Learning Without Image Regularization
## Introduction
Concept learning aims to enable T2I models to generate specific concepts and is widely used in the community. Nevertheless, the process of concept learning often involves model fine-tuning, which in turn brings the potential risk of language drift. Language drift typically manifests itself as degraded knowledge within the T2I model and a diminished editability for the target concept.
Our observation indicates that the commonly employed regularization method, which involves image regularization through the introduction of an image-caption subset, has certain limitations in addressing language drift.
Consequently, we introduce \textbf{TextReg}, a method designed to explicitly regulate model fine-tuning through the exclusive use of text prompts. Furthermore, we draw attention to the challenge of semantic drift that arises when integrating visual concepts into textual tokens and propose use identifier-mask to alleviate this issue. 

## Datasets


## Code Description 

### 1. Fine-tuning Stable Diffusion using Text-Regularization.
```
```

## Acknowledgements

This project is built upon the repository [lora](https://github.com/cloneofsimo/lora) and [diffusers](https://github.com/huggingface/diffusers). Special thanks to the contributors.

## Requirements

