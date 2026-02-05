# ASGARD Benchmark

> **Microsoft Responsible AI Transparency Documentation for Research**
>
> Code Release Readme Template — T&R Version: 8/28/2025

## Overview

ASGARD is a research benchmark for testing how well models can look at images, plan actions, and adjust when assumptions change in a simulated 3D home environment.

Unlike many embodied AI benchmarks, ASGARD tries to focus on visual reasoning and plan changes, not navigation or low‑level control.

ASGARD includes 108 task instances across 12 task types. Tasks are varied by things like object cleanliness, object placement, and scene setup. That means the same instruction can require different action sequences depending on what the agent sees.

### What Can the ASGARD Benchmark Do

ASGARD is meant to help researchers study whether models can:

- Use visual input to choose actions
- Notice when an assumption is wrong (e.g., something is dirty or closed)
- Update their plan based on what they see

ASGARD is built on AI2Thor and adds a higher-level action layer (for example, ASGARD handles navigation so we can focus on reasoning).

Agents interact using actions like `FIND`, `OPEN`, `PICKUP`, `PUT`, `CLEAN`, `TOGGLE_ON`, `TOGGLE_OFF`, and they only get simple success/failure feedback.

A detailed discussion of ASGARD, including how it was developed and evaluated, can be found in our paper at: [TODO: add link to paper].

### Intended Uses

ASGARD is intended for research evaluation of vision-language models and multimodal agents on visually grounded planning in simulation.

We are sharing ASGARD to support reproducible experiments and to encourage further work in this area.

This benchmark is intended for researchers and developers who can judge model outputs and results carefully before using them to make decisions.

ASGARD was designed and tested with English-language task instructions; results in other languages may differ.

### Out-of-Scope Uses

ASGARD is not meant to measure real-world robotics ability, navigation skill, or low-level manipulation control.

We do not recommend using ASGARD in commercial or real-world applications without more testing and development; it is being released for research purposes.

ASGARD was not designed or evaluated for all downstream uses. Anyone using it should evaluate accuracy, safety, and fairness for their specific use case.

Without further testing and development, ASGARD should not be used in sensitive domains where errors could lead to harm or impact someone's legal, financial, or life opportunities.

We do not recommend using ASGARD in situations related with high-risk decision making (e.g., law enforcement, legal, finance, or healthcare).

## How to Get Started

1. Get the code from the release repository: https://github.com/microsoft/AsgardBench
2. Follow the setup instructions in the README

## Evaluation

ASGARD is designed so that the same instruction can require different action sequences depending on visually observable state (for example, whether an object is dirty or a container is closed).

To check that ASGARD is actually testing visual grounding (and not just text priors), we ran ablations that reduce or remove visual input and observed a large drop in task success, which suggests the benchmark meaningfully depends on vision.

Full experimental details and results are in our paper.

## Limitations

- ASGARD is research software. More testing is needed before using it for real-world applications.
- ASGARD was designed and tested in English; performance in other languages may vary.
- Models may generate incorrect or made-up content. Users are responsible for checking outputs, and decisions should not be made based only on model outputs.
- ASGARD focuses on reasoning and plan repair under simplified interaction rules. Because it abstracts away navigation and low-level manipulation, results may not transfer to full embodied autonomy settings.
- ASGARD tasks are limited to household scenes (kitchens, living rooms, bathrooms) and the benchmark's specific task types and variations.
- ASGARD includes abstractions like `FIND` (navigation handled by the benchmark) and `PUT` behavior that can simplify placement compared to other environments. As such, it does not evaluate low-level navigation and manipulation ability.
- We have not done a systematic security review for vulnerabilities like indirect prompt injection. Anyone integrating ASGARD into a larger system should harden that system appropriately.

## Best Practices

Use ASGARD as a research benchmark, and avoid making real‑world deployment claims based on ASGARD results alone.

### For Reproducible Comparisons

- Use the provided fixed test set when comparing models, so results are reproducible.
- Do not change the evaluation prompt if your goal is to compare scores across models or reproduce published numbers. Even small prompt edits can change outcomes and make results harder to compare. If you do change the prompt, treat it as a separate benchmark variant and publish the full prompt alongside your results.
- When publishing results, include the model version and decoding settings (for example, temperature, max tokens, etc).
- Repeat runs when possible. Benchmark results can vary across runs; if you can, run each model more than once and report an average and standard deviation.

### Responsible AI Resources

We strongly encourage users to use LLMs/MLLMs that support robust Responsible AI mitigations, such as Azure Open AI (AOAI) services. Such services continually update their safety and RAI mitigations with the latest industry standards for responsible use.

For more on AOAI's best practices when employing foundation models for scripts and applications:

- [What is Azure AI Content Safety?](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview)
- [Overview of Responsible AI practices for Azure OpenAI models](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview)
- [Azure OpenAI Transparency Note](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note)
- [OpenAI's Usage policies](https://openai.com/policies/usage-policies)
- [Azure OpenAI's Code of Conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct)

It is the user's responsibility to ensure that the use of ASGARD complies with relevant data protection regulations and organizational guidelines.

## License

MIT License

Nothing disclosed here, including the Out of Scope Uses section, should be interpreted as or deemed a restriction or modification to the license the code is released under.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## Contact

This research was conducted by members of [Microsoft Research](https://www.microsoft.com/en-us/research/). We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please contact:

- Andrea Tupini ([andreatupini@microsoft.com](mailto:andreatupini@microsoft.com))
- Lars Liden ([Lars.Liden@microsoft.com](mailto:Lars.Liden@microsoft.com))

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.
