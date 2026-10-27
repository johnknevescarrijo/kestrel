//! ChatML prompt formatting utilities for standard, tool-augmented, and structured completions.

use crate::types::ChatMessage;
use kestrel_tools::{format_tools_prompt, ToolDefinition};

/// Formats a series of chat messages into a ChatML prompt template.
pub fn format_chatml_prompt(messages: &[ChatMessage]) -> String {
    format_chatml_prompt_with_tools(messages, None)
}

/// Formats chat messages and injects structured output instructions into the system prompt.
pub fn format_chatml_prompt_with_structured(
    messages: &[ChatMessage],
    mode: &kestrel_tools::StructuredOutputMode,
) -> String {
    let instruction = kestrel_tools::format_structured_output_prompt(mode);
    let mut prompt = String::new();
    let has_system = messages.iter().any(|m| m.role == "system");

    if !has_system {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(&instruction);
        prompt.push_str("<|im_end|>\n");
    }

    for msg in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&msg.role);
        prompt.push('\n');
        prompt.push_str(&msg.content);
        if msg.role == "system" {
            prompt.push_str("\n\n");
            prompt.push_str(&instruction);
        }
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Formats chat messages and optionally injects tool definitions into the ChatML prompt.
pub fn format_chatml_prompt_with_tools(
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
) -> String {
    let mut prompt = String::new();
    let has_system = messages.iter().any(|m| m.role == "system");

    if let Some(tool_list) = tools {
        if !tool_list.is_empty() && !has_system {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(&format_tools_prompt(tool_list));
            prompt.push_str("<|im_end|>\n");
        }
    }

    for msg in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&msg.role);
        prompt.push('\n');
        prompt.push_str(&msg.content);
        if msg.role == "system" {
            if let Some(tool_list) = tools {
                if !tool_list.is_empty() {
                    prompt.push_str("\n\n");
                    prompt.push_str(&format_tools_prompt(tool_list));
                }
            }
        }
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_chatml_prompt() {
        let messages = vec![
            ChatMessage::new("system", "You are helpful."),
            ChatMessage::new("user", "Hello!"),
        ];

        let prompt = format_chatml_prompt(&messages);
        let expected = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n";
        assert_eq!(prompt, expected);
    }
}
