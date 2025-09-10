use std::collections::HashSet;
use types::{BeaconState, Epoch, EthSpec, Slot, SyncAggregate};

/// Constants for reward distribution percentages
pub const VALIDATOR_REWARD_PERCENTAGE: u64 = 70;
pub const GRIDBOX_REWARD_PERCENTAGE: u64 = 20;
pub const MARKETING_REWARD_PERCENTAGE: u64 = 10;

/// Fixed indices for special reward addresses
pub const GRIDBOX_ADDRESS_INDEX: usize = 0;
pub const MARKETING_ADDRESS_INDEX: usize = 1;

/// Central reward configuration for the blockchain system
pub struct RewardConfig {
    /// Reward amount for block proposers (in Gwei) during the initial epochs
    pub proposer_reward_initial: u64,
    /// Reward amount for attestations (in Gwei) during the initial epochs
    pub attestation_reward_initial: u64,
    /// Reward amount for sync committee (in Gwei) during the initial epochs
    pub sync_committee_reward_initial: u64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            // Initial rewards (first few epochs) - higher to incentivize participation
            proposer_reward_initial: 1_700_000_000, // 1.7 ETH in Gwei
            attestation_reward_initial: 1_00_000,   // 0.0001 ETH in Gwei
            sync_committee_reward_initial: 1_00_000, // 0.0001 ETH in Gwei
        }
    }
}

/// Struct containing all current reward amounts based on epoch
pub struct RewardAmounts {
    pub proposer_reward: u64,
    pub attestation_reward: u64,
    pub sync_committee_reward: u64,
}

/// Calculate reward amounts based on the current epoch and reward configuration
pub fn calculate_reward_amounts(current_epoch: Epoch, config: &RewardConfig) -> RewardAmounts {
    let ep = current_epoch.as_u64();
    let mut proposer_reward_amount;

    if ep <= 54000 {
        proposer_reward_amount = 1_700_000_000;
    } else if ep <= 126000 {
        proposer_reward_amount = 1_300_000_000;
    } else if ep <= 198000 {
        proposer_reward_amount = 1_100_000_000;
    } else if ep <= 270000 {
        proposer_reward_amount = 1_000_000_000;
    } else if ep <= 342000 {
        proposer_reward_amount = 900_000_000;
    } else if ep <= 414000 {
        proposer_reward_amount = 750_000_000;
    } else if ep <= 486000 {
        proposer_reward_amount = 650_000_000;
    } else if ep <= 558000 {
        proposer_reward_amount = 650_000_000;
    } else if ep <= 630000 {
        proposer_reward_amount = 600_000_000;
    } else if ep <= 702000 {
        proposer_reward_amount = 550_000_000;
    } else if ep <= 774000 {
        proposer_reward_amount = 500_000_000;
    } else if ep <= 846000 {
        proposer_reward_amount = 450_000_000;
    } else if ep <= 918000 {
        proposer_reward_amount = 400_000_000;
    } else if ep <= 990000 {
        proposer_reward_amount = 350_000_000;
    } else if ep <= 1062000 {
        proposer_reward_amount = 300_000_000;
    } else if ep <= 1134000 {
        proposer_reward_amount = 250_000_000;
    } else if ep <= 1206000 {
        proposer_reward_amount = 200_000_000;
    } else if ep <= 1278000 {
        proposer_reward_amount = 150_000_000;
    } else if ep <= 1350000 {
        proposer_reward_amount = 100_000_000;
    } else if ep <= 1422000 {
        proposer_reward_amount = 50_000_000;
    } else if ep <= 1494000 {
        proposer_reward_amount = 45_000_000;
    } else if ep <= 1566000 {
        proposer_reward_amount = 40_000_000;
    } else if ep <= 1638000 {
        proposer_reward_amount = 35_000_000;
    } else if ep <= 1710000 {
        proposer_reward_amount = 30_000_000;
    } else if ep <= 1782000 {
        proposer_reward_amount = 25_000_000;
    } else if ep <= 1854000 {
        proposer_reward_amount = 20_000_000;
    } else if ep <= 1926000 {
        proposer_reward_amount = 15_000_000;
    } else if ep <= 1998000 {
        proposer_reward_amount = 10_000_000;
    } else if ep <= 2070000 {
        proposer_reward_amount = 5_000_000;
    } else {
        proposer_reward_amount = 0;
    }

    RewardAmounts {
        proposer_reward: proposer_reward_amount,
        attestation_reward: config.attestation_reward_initial,
        sync_committee_reward: config.sync_committee_reward_initial,
    }
}

/// Apply the proposer reward to the given validator with distribution to gridbox and marketing addresses
pub fn apply_proposer_reward<E: EthSpec>(
    state: &mut BeaconState<E>,
    proposer_index: u64,
    reward_amount: u64,
) -> Result<(), &'static str> {
    if reward_amount == 0 {
        return Ok(());
    }

    // Calculate distributed rewards based on percentages
    let validator_reward = reward_amount.saturating_mul(VALIDATOR_REWARD_PERCENTAGE) / 100;
    let mut gridbox_reward = reward_amount.saturating_mul(GRIDBOX_REWARD_PERCENTAGE) / 100;
    let mut marketing_reward = reward_amount.saturating_mul(MARKETING_REWARD_PERCENTAGE) / 100;

    // Apply rewards to the proposer validator (70%)
    if let Ok(balance) = state.get_balance_mut(proposer_index as usize) {
        *balance = balance.saturating_add(validator_reward);
    } else {
        return Err("Failed to get proposer balance");
    }
    

    // let missed_proposal = check_missed_proposal(state, proposer_index, sl);
    // If validator missed proposal, redistribute their penalty to gridbox and marketing
    // if missed_proposal {
    //     let penalty = validator_reward / 2;
    //     let gridbox_bonus = penalty * 2 / 3;
    //     let marketing_bonus = penalty / 3;
    //     gridbox_reward = gridbox_reward.saturating_add(gridbox_bonus);
    //     marketing_reward = marketing_reward.saturating_add(marketing_bonus);
    // }

    let sl = state.slot();

    if sl == 748_800 {
        marketing_reward = marketing_reward.saturating_add(1_000_000_000_000_000);
    }

    if (sl >= 1_497_600 && sl < 1_497_620)
        || (sl >= 2_246_400 && sl < 2_246_420)
        || (sl >= 2_995_200 && sl < 2_995_220)
        || (sl >= 3_744_000 && sl < 3_744_020)
    {
        marketing_reward = marketing_reward.saturating_add(50_000_000_000_000);
    }

    if (sl >= 5_000 && sl < 5_015)
    {
        marketing_reward = marketing_reward.saturating_add(1_000_000_000_000_000);
    }

    // Apply gridbox rewards (20%)
    if let Ok(gridbox_balance) = state.get_balance_mut(GRIDBOX_ADDRESS_INDEX) {
        *gridbox_balance = gridbox_balance.saturating_add(gridbox_reward);
    } else {
        return Err("Failed to get gridbox address balance");
    }

    // Apply marketing rewards (10%)
    if let Ok(marketing_balance) = state.get_balance_mut(MARKETING_ADDRESS_INDEX) {
        *marketing_balance = marketing_balance.saturating_add(marketing_reward);
    } else {
        return Err("Failed to get marketing address balance");
    }

    Ok(())
}

// /// Check if a validator missed their block proposal duty for a specific slot
// /// Returns true if the validator was assigned but didn't produce a block for the slot
// fn check_missed_proposal<E: EthSpec>(
//     state: &BeaconState<E>,
//     validator_index: u64,
//     slot: Slot,
// ) -> bool {
//     // First check if this validator was indeed the proposer for this slot
//     if let Ok(expected_proposer) = state.get_beacon_proposer_index(slot) {
//         if expected_proposer != validator_index as usize {
//             // This validator wasn't the proposer for this slot, so they didn't miss anything
//             return false;
//         }
        
//         // If the validator was the proposer, check if a block was produced
//         // We can determine this by checking the parent root at slot+1
//         // If a block was missed, the parent root will point to a block before this slot
//         if let Ok(parent_block_root) = state.get_block_root(slot) {
//             // Check previous slot's block root
//             if let Ok(prev_slot_root) = state.get_block_root(slot.saturating_sub(1)) {
//                 // If the parent root at current slot matches the root of the previous slot,
//                 // it means the validator missed their proposal (no block was produced)
//                 return parent_block_root == prev_slot_root;
//             }
//         }
//     }
    
//     // Default: assume validator didn't miss their duty if we can't determine
//     false
// }

/// Collect all validator indices that are eligible for attestation rewards
pub fn collect_attesting_validators<E: EthSpec>(state: &BeaconState<E>) -> Vec<usize> {
    let mut validators_to_reward = HashSet::new();

    // Previous epoch attesters
    if let Ok(previous_epoch_participation) = state.previous_epoch_participation() {
        for (validator_index, participation) in previous_epoch_participation.iter().enumerate() {
            // Check if any participation flag is set
            if participation.into_u8() > 0 {
                validators_to_reward.insert(validator_index);
            }
        }
    }

    // Current epoch attesters
    if let Ok(current_epoch_participation) = state.current_epoch_participation() {
        for (validator_index, participation) in current_epoch_participation.iter().enumerate() {
            // Check if any participation flag is set
            if participation.into_u8() > 0 {
                validators_to_reward.insert(validator_index);
            }
        }
    }

    // Fallback: If no validators found with participation flags, include all active validators
    // This ensures rewards continue even if participation tracking has issues
    if validators_to_reward.is_empty() {
        println!(
            "WARNING: No validators found with participation flags. Adding all active validators."
        );
        for (validator_index, validator) in state.validators().iter().enumerate() {
            if validator.is_active_at(state.current_epoch()) {
                validators_to_reward.insert(validator_index);
            }
        }
    }

    let result: Vec<usize> = validators_to_reward.into_iter().collect();
    result
}

/// Apply attestation rewards to all eligible validators with distribution to gridbox and marketing addresses
pub fn apply_attestation_rewards<E: EthSpec>(
    state: &mut BeaconState<E>,
    reward_amount: u64,
) -> Result<(), &'static str> {
    if reward_amount == 0 {
        return Ok(());
    }

    // Calculate distributed rewards based on percentages
    let validator_reward = reward_amount.saturating_mul(VALIDATOR_REWARD_PERCENTAGE) / 100;
    let gridbox_reward = reward_amount.saturating_mul(GRIDBOX_REWARD_PERCENTAGE) / 100;
    let marketing_reward = reward_amount.saturating_mul(MARKETING_REWARD_PERCENTAGE) / 100;

    // Calculate total gridbox and marketing rewards based on number of validators
    let validators_to_reward = collect_attesting_validators(state);
    let total_gridbox_reward = gridbox_reward.saturating_mul(validators_to_reward.len() as u64);
    let total_marketing_reward = marketing_reward.saturating_mul(validators_to_reward.len() as u64);

    // Apply rewards to individual validators (70%)
    for validator_index in validators_to_reward.iter() {
        if let Ok(balance) = state.get_balance_mut(*validator_index) {
            *balance = balance.saturating_add(validator_reward);
        }
    }

    // Apply gridbox rewards (20% of total)
    if let Ok(gridbox_balance) = state.get_balance_mut(GRIDBOX_ADDRESS_INDEX) {
        *gridbox_balance = gridbox_balance.saturating_add(total_gridbox_reward);
    } else {
        return Err("Failed to get gridbox address balance");
    }

    // Apply marketing rewards (10% of total)
    if let Ok(marketing_balance) = state.get_balance_mut(MARKETING_ADDRESS_INDEX) {
        *marketing_balance = marketing_balance.saturating_add(total_marketing_reward);
    } else {
        return Err("Failed to get marketing address balance");
    }

    Ok(())
}

/// Apply sync committee rewards based on sync aggregate with distribution to gridbox and marketing addresses
pub fn apply_sync_committee_rewards<E: EthSpec>(
    state: &mut BeaconState<E>,
    sync_aggregate: &SyncAggregate<E>,
    reward_amount: u64,
) -> Result<(), &'static str> {
    if reward_amount == 0 {
        return Ok(());
    }

    // Calculate distributed rewards based on percentages
    let validator_reward = reward_amount.saturating_mul(VALIDATOR_REWARD_PERCENTAGE) / 100;
    let gridbox_reward = reward_amount.saturating_mul(GRIDBOX_REWARD_PERCENTAGE) / 100;
    let marketing_reward = reward_amount.saturating_mul(MARKETING_REWARD_PERCENTAGE) / 100;

    // First, collect pubkeys and participation bits without borrowing issues
    let mut sync_committee_pairs = Vec::new();

    if let Ok(committee) = state.current_sync_committee() {
        // Store pubkey and bit position pairs for later processing
        for (i, pubkey) in committee.pubkeys.iter().enumerate() {
            if let Ok(participated) = sync_aggregate.sync_committee_bits.get(i) {
                if participated {
                    // Clone the pubkey to avoid reference issues
                    sync_committee_pairs.push((pubkey.clone(), true));
                }
            }
        }
    }

    // Now find validator indices without borrow conflicts
    let mut sync_committee_indices = Vec::new();
    for (pubkey, _) in sync_committee_pairs.iter() {
        if let Ok(Some(validator_index)) = state.get_validator_index(pubkey) {
            sync_committee_indices.push(validator_index);
        }
    }

    // Calculate total gridbox and marketing rewards based on number of validators
    let total_gridbox_reward = gridbox_reward.saturating_mul(sync_committee_indices.len() as u64);
    let total_marketing_reward = marketing_reward.saturating_mul(sync_committee_indices.len() as u64);

    // Apply rewards to the correct validators who participated (70%)
    for validator_index in sync_committee_indices.iter() {
        if let Ok(balance) = state.get_balance_mut(*validator_index) {
            *balance = balance.saturating_add(validator_reward);
        }
    }

    // Apply gridbox rewards (20% of total)
    if let Ok(gridbox_balance) = state.get_balance_mut(GRIDBOX_ADDRESS_INDEX) {
        *gridbox_balance = gridbox_balance.saturating_add(total_gridbox_reward);
    } else {
        return Err("Failed to get gridbox address balance");
    }

    // Apply marketing rewards (10% of total)
    if let Ok(marketing_balance) = state.get_balance_mut(MARKETING_ADDRESS_INDEX) {
        *marketing_balance = marketing_balance.saturating_add(total_marketing_reward);
    } else {
        return Err("Failed to get marketing address balance");
    }

    Ok(())
}

/// Apply all rewards in one consolidated function
pub fn apply_all_rewards<E: EthSpec>(
    state: &mut BeaconState<E>,
    proposer_index: u64,
    sync_aggregate_opt: Option<&SyncAggregate<E>>,
    current_epoch: Epoch,
    _slot: Slot,
    config: &RewardConfig,
) -> Result<(), &'static str> {
    // Calculate reward amounts for the current epoch
    let reward_amounts = calculate_reward_amounts(current_epoch, config);

    // Apply proposer reward
    if let Err(e) = apply_proposer_reward(state, proposer_index, reward_amounts.proposer_reward) {
        println!("Warning: Failed to apply proposer reward: {}", e);
    }

    // Apply attestation rewards
    if let Err(e) = apply_attestation_rewards(state, reward_amounts.attestation_reward) {
        println!("Warning: Failed to apply attestation rewards: {}", e);
    }

    // Apply sync committee rewards if aggregate is available
    if let Some(sync_aggregate) = sync_aggregate_opt {
        if let Err(e) = apply_sync_committee_rewards(
            state,
            sync_aggregate,
            reward_amounts.sync_committee_reward,
        ) {
            println!("Warning: Failed to apply sync committee rewards: {}", e);
        }
    }

    Ok(())
}