use crate::config::WorkloadConfig;

/// Fixed benchmark workloads with deterministic, reproducible prompts.
pub fn default_workloads() -> Vec<WorkloadConfig> {
    vec![
        WorkloadConfig {
            id: "summarize".to_string(),
            label: "Summarization".to_string(),
            system: Some("You are a helpful assistant. Be concise.".to_string()),
            prompt: SUMMARIZE_PROMPT.to_string(),
            max_tokens: 256,
        },
        WorkloadConfig {
            id: "code".to_string(),
            label: "Code Generation".to_string(),
            system: Some("You are an expert programmer. Write clean, correct code.".to_string()),
            prompt: CODE_PROMPT.to_string(),
            max_tokens: 512,
        },
        WorkloadConfig {
            id: "assistant".to_string(),
            label: "Generic Assistant".to_string(),
            system: Some("You are a helpful, harmless, and honest assistant.".to_string()),
            prompt: ASSISTANT_PROMPT.to_string(),
            max_tokens: 256,
        },
    ]
}

/// Fixed document for summarization – length ~600 tokens.
const SUMMARIZE_PROMPT: &str = r#"Summarize the following article in three concise bullet points.

Article:
The field of artificial intelligence has undergone a remarkable transformation over the past decade.
Initially dominated by rule-based systems and classical machine learning algorithms, the landscape
shifted dramatically with the rise of deep learning in the early 2010s. The introduction of
convolutional neural networks revolutionized computer vision tasks, achieving superhuman performance
on image recognition benchmarks by 2015. Meanwhile, recurrent neural networks and later transformer
architectures fundamentally changed how machines process sequential data such as text.

The transformer architecture, introduced in the seminal paper "Attention Is All You Need" in 2017,
laid the groundwork for a new generation of language models. Pre-training large models on vast
corpora of text and then fine-tuning them for specific tasks proved to be an extraordinarily
effective paradigm. Models such as BERT, GPT-2, and their successors demonstrated that scale
correlates strongly with capability across diverse natural language understanding and generation tasks.

By 2022 the field had produced models with hundreds of billions of parameters capable of performing
complex reasoning, writing poetry, generating code, and engaging in extended multi-turn dialogues.
These capabilities raised substantial questions about safety, alignment, economic disruption, and
the philosophical nature of understanding in artificial systems. Governments, academic institutions,
and technology companies began debating regulatory frameworks even as the pace of research
continued to accelerate.

Open-source efforts also gained significant momentum, with organizations releasing model weights
under permissive licenses, enabling a global community of researchers and practitioners to study,
fine-tune, and deploy capable language models on consumer hardware. Quantization techniques further
democratized access by allowing multi-billion parameter models to run on machines with modest GPU
memory or even on CPUs alone, expanding the potential user base to millions of developers worldwide.

Summarize the above in three bullet points:"#;

/// Fixed coding task with a clear specification.
const CODE_PROMPT: &str = r#"Write a Python function called `merge_sorted_arrays` that merges two sorted arrays into a single sorted array without using the built-in `sorted()` function or `list.sort()`. The function should:
- Accept two lists of comparable elements
- Return a new sorted list containing all elements from both inputs
- Run in O(n + m) time where n and m are the lengths of the inputs
- Include a brief docstring

After the function, add three assert statements that test it with concrete inputs. Write only the Python code, no explanation outside of the docstring."#;

/// Generic assistant prompt exercising multi-step reasoning.
const ASSISTANT_PROMPT: &str = r#"I need to plan a 7-day solo hiking trip through a mountainous region. I have intermediate hiking experience, a tent, and a budget of roughly $500 beyond transportation. The trip starts and ends at the same trailhead. List the five most important things I should organize or purchase before departure, briefly explain the reason for each, and then suggest one specific piece of gear that is often overlooked by intermediate hikers. Keep the response focused and practical."#;
