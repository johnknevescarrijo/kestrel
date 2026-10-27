//! File-based cold storage implementation for session persistence.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use tracing::{debug, warn};

use crate::error::Error;
use crate::session::{SessionId, SessionMetadata, SessionSnapshot};
use crate::traits::SessionStore;

const SESSION_FILE_EXTENSION: &str = "kestrel";

/// Session persistence store backed by disk files serialized with `bincode`.
///
/// Each session is stored at `<base_dir>/<session_id>.kestrel`. Writes are performed
/// atomically via a temporary file and atomic rename.
#[derive(Debug, Clone)]
pub struct FileSessionStore {
    base_dir: PathBuf,
}

impl FileSessionStore {
    /// Creates a new `FileSessionStore` saving sessions under `base_dir`.
    ///
    /// The base directory is created if it does not already exist.
    pub fn new(base_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = base_dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir).map_err(|err| Error::SessionStorageFailed {
            path: dir.display().to_string(),
            reason: format!("failed to create base directory: {err}"),
        })?;
        Ok(Self { base_dir: dir })
    }

    /// Returns the base directory where sessions are stored.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Validates the session ID against path traversal and returns the target file path.
    fn session_file_path(&self, id: &SessionId) -> Result<PathBuf, Error> {
        validate_session_id(id.as_str())?;
        let filename = format!("{}.{SESSION_FILE_EXTENSION}", id.as_str());
        Ok(self.base_dir.join(filename))
    }
}

/// Checks if a session ID matches any reserved Windows device filename.
pub fn is_windows_reserved_name(id: &str) -> bool {
    let upper = id.to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
    )
}

/// Validates that a session ID does not contain path traversal characters or Windows reserved device names.
fn validate_session_id(id: &str) -> Result<(), Error> {
    if id.is_empty() {
        return Err(Error::InvalidSessionId {
            id: id.to_string(),
            reason: "session ID cannot be empty".to_string(),
        });
    }

    if is_windows_reserved_name(id) {
        return Err(Error::InvalidSessionId {
            id: id.to_string(),
            reason: "session ID cannot use Windows reserved device names".to_string(),
        });
    }

    let is_valid = id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');

    if !is_valid || id.contains("..") {
        return Err(Error::InvalidSessionId {
            id: id.to_string(),
            reason:
                "contains forbidden path characters; must only contain alphanumeric, '-', or '_'"
                    .to_string(),
        });
    }

    Ok(())
}

#[async_trait]
impl SessionStore for FileSessionStore {
    async fn save(&self, snapshot: &SessionSnapshot) -> Result<(), Error> {
        let target_path = self.session_file_path(&snapshot.metadata.id)?;
        let base_dir = self.base_dir.clone();
        let bytes =
            bincode::serialize(snapshot).map_err(|err| Error::SessionSerializationFailed {
                reason: format!(
                    "failed to serialize snapshot for '{}': {err}",
                    snapshot.metadata.id
                ),
            })?;

        let id_str = snapshot.metadata.id.as_str().to_string();

        tokio::task::spawn_blocking(move || -> Result<(), Error> {
            let temp_path = base_dir.join(format!(".tmp_{id_str}.{SESSION_FILE_EXTENSION}"));

            let mut file = File::create(&temp_path).map_err(|err| Error::SessionStorageFailed {
                path: temp_path.display().to_string(),
                reason: format!("failed to create temporary file: {err}"),
            })?;

            file.write_all(&bytes)
                .map_err(|err| Error::SessionStorageFailed {
                    path: temp_path.display().to_string(),
                    reason: format!("failed to write data: {err}"),
                })?;

            file.sync_all().map_err(|err| Error::SessionStorageFailed {
                path: temp_path.display().to_string(),
                reason: format!("failed to sync temporary file: {err}"),
            })?;

            #[cfg(windows)]
            if target_path.exists() {
                let _ = fs::remove_file(&target_path);
            }

            if let Err(err) = fs::rename(&temp_path, &target_path) {
                if target_path.exists() {
                    let _ = fs::remove_file(&target_path);
                    fs::rename(&temp_path, &target_path).map_err(|retry_err| {
                        Error::SessionStorageFailed {
                            path: target_path.display().to_string(),
                            reason: format!(
                                "failed to rename temporary file to target: {retry_err}"
                            ),
                        }
                    })?;
                } else {
                    return Err(Error::SessionStorageFailed {
                        path: target_path.display().to_string(),
                        reason: format!("failed to rename temporary file to target: {err}"),
                    });
                }
            }

            debug!(path = %target_path.display(), "Session snapshot saved successfully");
            Ok(())
        })
        .await
        .map_err(|err| Error::Internal {
            reason: format!("storage save blocking task failed: {err}"),
        })?
    }

    async fn load(&self, id: &SessionId) -> Result<Option<SessionSnapshot>, Error> {
        let path = self.session_file_path(id)?;
        if !path.exists() {
            return Ok(None);
        }

        tokio::task::spawn_blocking(move || -> Result<Option<SessionSnapshot>, Error> {
            let bytes = match fs::read(&path) {
                Ok(b) => b,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(err) => {
                    return Err(Error::SessionStorageFailed {
                        path: path.display().to_string(),
                        reason: format!("failed to read session file: {err}"),
                    })
                }
            };

            let snapshot: SessionSnapshot =
                bincode::deserialize(&bytes).map_err(|err| Error::SessionSerializationFailed {
                    reason: format!(
                        "failed to deserialize session at '{}': {err}",
                        path.display()
                    ),
                })?;

            Ok(Some(snapshot))
        })
        .await
        .map_err(|err| Error::Internal {
            reason: format!("storage load blocking task failed: {err}"),
        })?
    }

    async fn delete(&self, id: &SessionId) -> Result<bool, Error> {
        let path = self.session_file_path(id)?;
        if !path.exists() {
            return Ok(false);
        }

        tokio::task::spawn_blocking(move || -> Result<bool, Error> {
            match fs::remove_file(&path) {
                Ok(_) => Ok(true),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
                Err(err) => Err(Error::SessionStorageFailed {
                    path: path.display().to_string(),
                    reason: format!("failed to delete session file: {err}"),
                }),
            }
        })
        .await
        .map_err(|err| Error::Internal {
            reason: format!("storage delete blocking task failed: {err}"),
        })?
    }

    async fn list(&self) -> Result<Vec<SessionMetadata>, Error> {
        let base_dir = self.base_dir.clone();

        tokio::task::spawn_blocking(move || -> Result<Vec<SessionMetadata>, Error> {
            let mut list = Vec::new();
            let entries = fs::read_dir(&base_dir).map_err(|err| Error::SessionStorageFailed {
                path: base_dir.display().to_string(),
                reason: format!("failed to read directory: {err}"),
            })?;

            for entry in entries {
                let entry = entry.map_err(|err| Error::SessionStorageFailed {
                    path: base_dir.display().to_string(),
                    reason: format!("failed reading dir entry: {err}"),
                })?;

                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) == Some(SESSION_FILE_EXTENSION) {
                    if let Ok(bytes) = fs::read(&path) {
                        if let Ok(snapshot) = bincode::deserialize::<SessionSnapshot>(&bytes) {
                            list.push(snapshot.metadata);
                        } else {
                            warn!(path = %path.display(), "Skipping corrupted session file");
                        }
                    }
                }
            }

            Ok(list)
        })
        .await
        .map_err(|err| Error::Internal {
            reason: format!("storage list blocking task failed: {err}"),
        })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_file_session_store_crud() {
        let dir = tempdir().unwrap();
        let store = FileSessionStore::new(dir.path()).unwrap();

        let id = SessionId::from("sess-100");
        let mut snapshot = SessionSnapshot {
            metadata: SessionMetadata::new(id.clone(), "test-model"),
            context_text: "System: start".to_string(),
            tokens: vec![10, 20],
            kv_state: vec![1, 2, 3],
        };
        snapshot.metadata.total_tokens = 2;

        // Initially does not exist
        assert_eq!(store.load(&id).await.unwrap(), None);

        // Save
        store.save(&snapshot).await.unwrap();

        // Load
        let loaded = store
            .load(&id)
            .await
            .unwrap()
            .expect("should find snapshot");
        assert_eq!(loaded.metadata.id, id);
        assert_eq!(loaded.context_text, "System: start");
        assert_eq!(loaded.tokens, vec![10, 20]);
        assert_eq!(loaded.kv_state, vec![1, 2, 3]);

        // List
        let list = store.list().await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, id);

        // Delete
        assert!(store.delete(&id).await.unwrap());
        assert_eq!(store.load(&id).await.unwrap(), None);
        assert!(!store.delete(&id).await.unwrap());
    }

    #[test]
    fn test_validate_session_id_security() {
        assert!(validate_session_id("valid-id_123").is_ok());
        assert!(validate_session_id("").is_err());
        assert!(validate_session_id("../escaped").is_err());
        assert!(validate_session_id("foo/bar").is_err());
        assert!(validate_session_id("foo\\bar").is_err());
        assert!(validate_session_id("null\0byte").is_err());
    }

    #[test]
    fn test_validate_session_id_windows_reserved_names() {
        assert!(is_windows_reserved_name("CON"));
        assert!(is_windows_reserved_name("con"));
        assert!(is_windows_reserved_name("Prn"));
        assert!(is_windows_reserved_name("aux"));
        assert!(is_windows_reserved_name("NUL"));
        assert!(is_windows_reserved_name("com1"));
        assert!(is_windows_reserved_name("COM9"));
        assert!(is_windows_reserved_name("lpt1"));
        assert!(is_windows_reserved_name("LPT9"));
        assert!(!is_windows_reserved_name("CONSOLE"));
        assert!(!is_windows_reserved_name("normal_session"));

        assert!(validate_session_id("CON").is_err());
        assert!(validate_session_id("aux").is_err());
        assert!(validate_session_id("com5").is_err());
        assert!(validate_session_id("lpt2").is_err());
    }
}
