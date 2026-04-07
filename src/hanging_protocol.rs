//! Hanging protocol support for automatic viewport layout and series assignment.
//!
//! Configured via `[[hanging_protocol]]` sections in `~/.config/sauhu/config.toml`.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A hanging protocol defines how viewports should be laid out and which
/// series should be assigned to each slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HangingProtocol {
    pub name: String,
    /// Layout name (e.g. "grid_4x2", "grid_2x2")
    pub layout: String,
    /// Higher priority wins when multiple protocols match
    #[serde(default)]
    pub priority: i32,
    /// Criteria for matching this protocol to a study
    #[serde(rename = "match")]
    pub match_criteria: ProtocolMatchCriteria,
    /// Slot assignment rules
    #[serde(default)]
    pub slot: Vec<SlotRule>,
}

/// Criteria for matching a protocol to a study.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolMatchCriteria {
    /// Required modality (e.g. "MR", "CT")
    pub modality: String,
    /// Keywords to match against study_description (case-insensitive, any match suffices)
    #[serde(default)]
    pub study_keywords: Vec<String>,
}

/// Rule for assigning a series to a viewport slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotRule {
    /// Viewport slot index (0-based)
    pub index: usize,
    /// Keywords to match against series_description (case-insensitive, any match = candidate)
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Keywords that disqualify a series from this slot
    #[serde(default)]
    pub exclude_keywords: Vec<String>,
    /// Prefer series with many images (>50), useful for 3D acquisitions
    #[serde(default)]
    pub prefer_3d: bool,
}

/// Runtime state for an active hanging protocol.
#[derive(Default)]
pub struct ProtocolState {
    /// The active protocol (None if no protocol matched)
    pub active_protocol: Option<HangingProtocol>,
    /// Mapping of slot_index -> series_uid for filled slots
    pub filled_slots: HashMap<usize, String>,
    /// Set of series UIDs that have been assigned to a slot
    pub assigned_series: HashSet<String>,
    /// Set to true when user manually loads a series, stopping auto-assignment
    pub user_overridden: bool,
}

impl ProtocolState {
    pub fn reset(&mut self) {
        self.active_protocol = None;
        self.filled_slots.clear();
        self.assigned_series.clear();
        self.user_overridden = false;
    }

    pub fn is_active(&self) -> bool {
        self.active_protocol.is_some()
    }
}

/// Find the best matching protocol for a study.
///
/// Returns the highest-priority protocol whose modality matches and at least
/// one study_keyword matches the study description (case-insensitive).
/// If study_keywords is empty, modality match alone is sufficient.
pub fn match_protocol<'a>(
    protocols: &'a [HangingProtocol],
    study_modality: &str,
    study_description: &str,
) -> Option<&'a HangingProtocol> {
    // DICOM modality can be multi-valued (e.g., "MR\SR"), split and check each
    let modality_parts: Vec<&str> = study_modality.split('\\').collect();
    let desc_lower = study_description.to_lowercase();

    let mut best: Option<&HangingProtocol> = None;

    for protocol in protocols {
        // Modality must match any part of the study modality
        let protocol_mod = protocol.match_criteria.modality.to_uppercase();
        let modality_matches = modality_parts
            .iter()
            .any(|part| part.to_uppercase() == protocol_mod);
        if !modality_matches {
            continue;
        }

        // If study_keywords is non-empty, at least one must match
        if !protocol.match_criteria.study_keywords.is_empty() {
            let any_match = protocol
                .match_criteria
                .study_keywords
                .iter()
                .any(|kw| desc_lower.contains(&kw.to_lowercase()));
            if !any_match {
                continue;
            }
        }

        // Pick highest priority
        if best.is_none_or(|b| protocol.priority > b.priority) {
            best = Some(protocol);
        }
    }

    best
}

/// Score a series against a slot rule.
///
/// Returns None if the series is excluded or has no keyword match.
/// Otherwise returns a score (higher = better fit).
pub fn score_series_for_slot(rule: &SlotRule, series_desc: &str, num_images: u32) -> Option<i32> {
    let desc_lower = series_desc.to_lowercase();

    // Check exclusions first
    for kw in &rule.exclude_keywords {
        if desc_lower.contains(&kw.to_lowercase()) {
            return None;
        }
    }

    // If no keywords defined, slot is for manual assignment only
    if rule.keywords.is_empty() {
        return None;
    }

    // Count keyword matches
    let hits: i32 = rule
        .keywords
        .iter()
        .filter(|kw| desc_lower.contains(&kw.to_lowercase()))
        .count() as i32;

    if hits == 0 {
        return None;
    }

    let mut score = hits * 10;

    // Bonus for prefer_3d with high image count
    if rule.prefer_3d && num_images > 50 {
        score += 5;
    }

    Some(score)
}

/// Information about a series needed for protocol assignment.
pub struct SeriesCandidate {
    pub series_uid: String,
    pub description: String,
    pub num_images: u32,
}

/// Try to assign a single series to the best unfilled slot.
///
/// Returns the slot index if assigned, None otherwise.
pub fn try_assign_series(state: &mut ProtocolState, candidate: &SeriesCandidate) -> Option<usize> {
    if state.user_overridden {
        return None;
    }

    let protocol = state.active_protocol.as_ref()?;

    // Skip if already assigned
    if state.assigned_series.contains(&candidate.series_uid) {
        return None;
    }

    let mut best_slot: Option<(usize, i32)> = None;

    for rule in &protocol.slot {
        // Skip filled slots
        if state.filled_slots.contains_key(&rule.index) {
            continue;
        }

        if let Some(score) =
            score_series_for_slot(rule, &candidate.description, candidate.num_images)
        {
            if best_slot.is_none_or(|(_, best_score)| score > best_score) {
                best_slot = Some((rule.index, score));
            }
        }
    }

    if let Some((slot_index, _)) = best_slot {
        state
            .filled_slots
            .insert(slot_index, candidate.series_uid.clone());
        state.assigned_series.insert(candidate.series_uid.clone());
        return Some(slot_index);
    }

    None
}

/// Assign all series to slots at once using greedy best-score-first matching.
///
/// Returns a list of (slot_index, series_uid) assignments.
pub fn assign_all_series(
    state: &mut ProtocolState,
    candidates: &[SeriesCandidate],
) -> Vec<(usize, String)> {
    if state.user_overridden {
        return Vec::new();
    }

    let protocol = match &state.active_protocol {
        Some(p) => p,
        None => return Vec::new(),
    };

    // Build all (slot_index, candidate_index, score) tuples
    let mut scored: Vec<(usize, usize, i32)> = Vec::new();
    for (ci, candidate) in candidates.iter().enumerate() {
        if state.assigned_series.contains(&candidate.series_uid) {
            continue;
        }
        for rule in &protocol.slot {
            if state.filled_slots.contains_key(&rule.index) {
                continue;
            }
            if let Some(score) =
                score_series_for_slot(rule, &candidate.description, candidate.num_images)
            {
                scored.push((rule.index, ci, score));
            }
        }
    }

    // Sort by score descending (greedy: best matches first)
    scored.sort_by(|a, b| b.2.cmp(&a.2));

    let mut used_slots: HashSet<usize> = HashSet::new();
    let mut used_candidates: HashSet<usize> = HashSet::new();
    let mut assignments: Vec<(usize, String)> = Vec::new();

    for (slot_index, ci, _score) in &scored {
        if used_slots.contains(slot_index) || used_candidates.contains(ci) {
            continue;
        }
        used_slots.insert(*slot_index);
        used_candidates.insert(*ci);

        let uid = candidates[*ci].series_uid.clone();
        state.filled_slots.insert(*slot_index, uid.clone());
        state.assigned_series.insert(uid.clone());
        assignments.push((*slot_index, uid));
    }

    assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_protocol() -> HangingProtocol {
        HangingProtocol {
            name: "MRI Brain".to_string(),
            layout: "grid_4x2".to_string(),
            priority: 100,
            match_criteria: ProtocolMatchCriteria {
                modality: "MR".to_string(),
                study_keywords: vec!["brain".to_string(), "head".to_string()],
            },
            slot: vec![
                SlotRule {
                    index: 0,
                    keywords: vec!["T1".to_string(), "3D".to_string()],
                    exclude_keywords: vec![],
                    prefer_3d: true,
                },
                SlotRule {
                    index: 1,
                    keywords: vec!["T2".to_string()],
                    exclude_keywords: vec!["FLAIR".to_string(), "STAR".to_string()],
                    prefer_3d: false,
                },
                SlotRule {
                    index: 2,
                    keywords: vec!["FLAIR".to_string()],
                    exclude_keywords: vec![],
                    prefer_3d: false,
                },
            ],
        }
    }

    #[test]
    fn test_match_protocol_modality_and_keyword() {
        let protocols = vec![make_protocol()];
        assert!(match_protocol(&protocols, "MR", "MRI Brain").is_some());
        assert!(match_protocol(&protocols, "MR", "HEAD CT").is_some()); // "head" matches
        assert!(match_protocol(&protocols, "CT", "MRI Brain").is_none()); // wrong modality
        assert!(match_protocol(&protocols, "MR", "Knee").is_none()); // no keyword match
    }

    #[test]
    fn test_match_protocol_multi_value_modality() {
        let protocols = vec![make_protocol()];
        // DICOM multi-value modality (e.g., "MR\SR") should match protocol with "MR"
        assert!(match_protocol(&protocols, "MR\\SR", "MRI Brain").is_some());
        assert!(match_protocol(&protocols, "SR\\MR", "Head scan").is_some());
        assert!(match_protocol(&protocols, "CT\\SR", "MRI Brain").is_none()); // no MR
    }

    #[test]
    fn test_match_protocol_priority() {
        let mut low = make_protocol();
        low.priority = 50;
        low.name = "Low".to_string();
        let mut high = make_protocol();
        high.priority = 200;
        high.name = "High".to_string();
        let protocols = vec![low, high];
        let matched = match_protocol(&protocols, "MR", "Brain MRI").unwrap();
        assert_eq!(matched.name, "High");
    }

    #[test]
    fn test_score_series_exclude() {
        let rule = &make_protocol().slot[1]; // T2, excludes FLAIR
        assert!(score_series_for_slot(rule, "T2 FLAIR", 30).is_none());
        assert!(score_series_for_slot(rule, "T2 TSE", 30).is_some());
    }

    #[test]
    fn test_score_series_prefer_3d() {
        let rule = &make_protocol().slot[0]; // T1 3D, prefer_3d
        let score_few = score_series_for_slot(rule, "T1 3D MPRAGE", 20).unwrap();
        let score_many = score_series_for_slot(rule, "T1 3D MPRAGE", 200).unwrap();
        assert!(score_many > score_few);
    }

    #[test]
    fn test_assign_all_series() {
        let protocol = make_protocol();
        let mut state = ProtocolState::default();
        state.active_protocol = Some(protocol);

        let candidates = vec![
            SeriesCandidate {
                series_uid: "1".to_string(),
                description: "T2 TSE".to_string(),
                num_images: 30,
            },
            SeriesCandidate {
                series_uid: "2".to_string(),
                description: "T1 3D MPRAGE".to_string(),
                num_images: 180,
            },
            SeriesCandidate {
                series_uid: "3".to_string(),
                description: "FLAIR".to_string(),
                num_images: 30,
            },
            SeriesCandidate {
                series_uid: "4".to_string(),
                description: "DWI".to_string(),
                num_images: 60,
            },
        ];

        let assignments = assign_all_series(&mut state, &candidates);
        assert_eq!(assignments.len(), 3); // T1->0, T2->1, FLAIR->2, DWI has no slot

        // Verify slot assignments
        assert_eq!(state.filled_slots.get(&0), Some(&"2".to_string())); // T1 3D
        assert_eq!(state.filled_slots.get(&1), Some(&"1".to_string())); // T2
        assert_eq!(state.filled_slots.get(&2), Some(&"3".to_string())); // FLAIR
    }
}
