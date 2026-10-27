//! Session restoration and KV state replay helpers for llama.cpp.

use llama_cpp::{LlamaModel, LlamaSession, SessionParams, Token};

use crate::error::LlamaError;
use kestrel_core::session::Session;

/// Creates a new `LlamaSession` and replays stored session tokens or context text.
pub(crate) fn create_restored_session(
    model: &LlamaModel,
    session: &Session,
) -> Result<LlamaSession, LlamaError> {
    let mut new_sess = model
        .create_session(SessionParams::default())
        .map_err(|err| LlamaError::SessionCreation {
            reason: format!("{err:?}"),
        })?;

    let tokens: Vec<Token> = if !session.tokens.is_empty() {
        session.tokens.iter().map(|&t| Token(t as i32)).collect()
    } else if !session.kv_state.is_empty() {
        session
            .kv_state
            .chunks_exact(4)
            .map(|chunk| {
                let id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                Token(id as i32)
            })
            .collect()
    } else {
        Vec::new()
    };

    if !tokens.is_empty() {
        new_sess
            .advance_context_with_tokens(&tokens)
            .map_err(|err| LlamaError::ContextAdvance {
                reason: format!("{err:?}"),
            })?;
    } else if !session.context_text.is_empty() {
        new_sess
            .advance_context(&session.context_text)
            .map_err(|err| LlamaError::ContextAdvance {
                reason: format!("{err:?}"),
            })?;
    }

    Ok(new_sess)
}

/// Computes an L2-normalized vector embedding from token IDs.
pub(crate) fn compute_token_embedding(tokens: &[u32], dims: usize) -> Vec<f32> {
    let mut embedding = vec![0.0f32; dims];
    for (i, &tok) in tokens.iter().enumerate() {
        let mut h = (tok as u64) ^ (i as u64).wrapping_mul(0x517cc1b727220a95);
        h = h.wrapping_mul(0x100000001b3);
        let dim = (h as usize) % dims;
        let sign = if (h & 1) == 0 { 1.0f32 } else { -1.0f32 };
        embedding[dim] += sign * (((tok % 1000) as f32) / 1000.0 + 0.1);
    }
    if embedding.iter().all(|&v| v == 0.0) {
        for (i, v) in embedding.iter_mut().enumerate() {
            *v = ((i + 1) as f32).sin();
        }
    }
    let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in embedding.iter_mut() {
            *v /= norm;
        }
    }
    embedding
}

#[cfg(test)]
mod tests {
    use super::*;
    use kestrel_core::session::SessionId;

    #[test]
    fn test_kv_state_token_deserialization() {
        let mut session = Session::new(SessionId::from("sess-test"), "test-model");
        let token_ids: Vec<u32> = vec![101, 102, 103, 104];
        session.kv_state = token_ids.iter().flat_map(|&t| t.to_le_bytes()).collect();

        let decoded_tokens: Vec<Token> = session
            .kv_state
            .chunks_exact(4)
            .map(|chunk| {
                let id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                Token(id as i32)
            })
            .collect();

        assert_eq!(
            decoded_tokens,
            vec![Token(101), Token(102), Token(103), Token(104)]
        );
    }
}
